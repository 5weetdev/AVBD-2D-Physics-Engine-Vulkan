// The C ABI of the 2D solver (avbd2d.h, ABI 3) on VkWorld2D + Scene2D.
//
// Shape of the layer: the host keeps slot tables (with free lists and generations) of bodies,
// shapes and joints. A created object is a record in its table and is "pending" until the next
// step_begin, which replays every pending body (with its shapes) and joint into the live world's
// authored Scene2D in one go and commits them (VkWorld2D::commitAuthored). Setters on a live
// object go to VkWorld2D's mutation queue; on a pending object they edit its record. Getters read
// VkWorld2D's host mirrors (the last completed step plus queued pose/velocity sets).
//
// Every entry point validates its handle or id first, so a null, never-created or destroyed world
// and a stale id are error codes and not crashes.


// Shapes and chains of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    // ----- Shape construction ----------------------------------------------------------------------
    AVBD2D_API Avbd2dPolygon avbd2d_make_box(float hx, float hy)
    {
        const Vec2 pts[4] = {Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy)};
        Avbd2dPolygon p = fillPolygon(pts, 4, 0.0f);
        p.centroid = Avbd2dVec2{0.0f, 0.0f};
        return p;
    }

    AVBD2D_API Avbd2dPolygon avbd2d_make_offset_box(float hx, float hy, Avbd2dVec2 center, Avbd2dRot rotation)
    {
        const float a = angleOf(rotation);
        const Vec2 pts[4] = {Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy)};
        Vec2 q[4];
        for (int i = 0; i < 4; i++)
            q[i] = V(center) + rotate(a, pts[i]);
        Avbd2dPolygon p = fillPolygon(q, 4, 0.0f);
        p.centroid = center;
        return p;
    }

    AVBD2D_API Avbd2dPolygon avbd2d_make_rounded_box(float hx, float hy, float radius)
    {
        Avbd2dPolygon p = avbd2d_make_box(hx, hy);
        p.radius = radius;
        return p;
    }

    AVBD2D_API Avbd2dResult avbd2d_make_polygon(const Avbd2dVec2 *points, int32_t count, float radius,
                                                Avbd2dPolygon *out)
    {
        NEED(points);
        NEED(out);
        if (count < 3 || count > AVBD2D_MAX_POLYGON_VERTICES || !(radius >= 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        Vec2 in[AVBD2D_MAX_POLYGON_VERTICES], hull[AVBD2D_MAX_POLYGON_VERTICES];
        for (int i = 0; i < count; i++)
        {
            if (!finite2(points[i]))
                return AVBD2D_ERR_INVALID_ARG;
            in[i] = V(points[i]);
        }
        const int n = Scene2D::computeHull(in, count, hull);
        if (n < 3)
            return AVBD2D_ERR_INVALID_ARG;
        *out = fillPolygon(hull, n, radius);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_make_proxy(const Avbd2dVec2 *points, int32_t count, float radius,
                                              Avbd2dShapeProxy *out)
    {
        NEED(points);
        NEED(out);
        if (count < 1 || count > AVBD2D_MAX_POLYGON_VERTICES || !(radius >= 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        *out = Avbd2dShapeProxy{};
        for (int i = 0; i < count; i++)
            out->points[i] = points[i];
        out->count = count;
        out->radius = radius;
        return AVBD2D_OK;
    }

    // Releases a shape record; a live shape keeps its solver id for the next step's events.
    static void retireShape(Avbd2dWorld *w, int slot)
    {
        const int solver = w->shapes.items[(size_t)slot].solver;
        if (solver >= 0 && solver < (int)w->solverShape.size())
        {
            tombstone(w, solver, shapeIdOf(w, slot));
            w->solverShape[(size_t)solver] = -1;
        }
        w->shapes.release(slot);
    }

    // A shape edit on a live body: its shape range is replaced at the next step.
    static void markReshape(Avbd2dWorld *w, int bodySlot)
    {
        BodyRec &b = w->bodies.items[(size_t)bodySlot];
        b.propsValid = false;
        markDirty(w, bodySlot);
        if (b.flushed && b.solver >= 0 && !b.reshape)
        {
            b.reshape = true;
            w->replaceBodies.push_back(bodySlot);
        }
    }

    // Adds a shape record to a body (a live body gets it at the next step).
    static Avbd2dResult addShapeRec(Avbd2dBodyId bid, const Avbd2dShapeDef *def, const ShapeRec &proto,
                                    Avbd2dShapeId *out_id)
    {
        BODY_CTX(bid);
        NEED(out_id);
        if (!validShapeDef(def))
            return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, true,
                       [&]() -> Avbd2dResult
                       {
                           const int s = w->shapes.acquire();
                           ShapeRec &r = w->shapes.items[(size_t)s];
                           const uint16_t gen = r.gen;
                           r = proto;
                           r.alive = true;
                           r.gen = gen;
                           r.body = slot;
                           r.def = *def;
                           b->shapes.push_back(s);
                           markReshape(w, slot);
                           *out_id = shapeIdOf(w, s);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_create_polygon_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                        const Avbd2dPolygon *polygon, Avbd2dShapeId *out_id)
    {
        NEED(polygon);
        if (polygon->count < 3 || polygon->count > AVBD2D_MAX_POLYGON_VERTICES || !(polygon->radius >= 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        Vec2 in[AVBD2D_MAX_POLYGON_VERTICES], hull[AVBD2D_MAX_POLYGON_VERTICES];
        for (int i = 0; i < polygon->count; i++)
        {
            if (!finite2(polygon->vertices[i]))
                return AVBD2D_ERR_INVALID_ARG;
            in[i] = V(polygon->vertices[i]);
        }
        if (Scene2D::computeHull(in, polygon->count, hull) < 3)
            return AVBD2D_ERR_INVALID_ARG;
        ShapeRec r;
        r.kind = SK_POLYGON;
        r.poly = *polygon;
        return addShapeRec(body, def, r, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_circle_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                       const Avbd2dCircle *circle, Avbd2dShapeId *out_id)
    {
        NEED(circle);
        if (!(circle->radius > 0.0f) || !finite2(circle->center))
            return AVBD2D_ERR_INVALID_ARG;
        ShapeRec r;
        r.kind = SK_CIRCLE;
        r.circle = *circle;
        return addShapeRec(body, def, r, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_capsule_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                        const Avbd2dCapsule *capsule, Avbd2dShapeId *out_id)
    {
        NEED(capsule);
        if (!(capsule->radius > 0.0f) || !finite2(capsule->center1) || !finite2(capsule->center2))
            return AVBD2D_ERR_INVALID_ARG;
        ShapeRec r;
        r.kind = SK_CAPSULE;
        r.capsule = *capsule;
        return addShapeRec(body, def, r, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_segment_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                        const Avbd2dSegment *segment, Avbd2dShapeId *out_id)
    {
        NEED(segment);
        if (!finite2(segment->point1) || !finite2(segment->point2))
            return AVBD2D_ERR_INVALID_ARG;
        ShapeRec r;
        r.kind = SK_SEGMENT;
        r.seg = *segment;
        return addShapeRec(body, def, r, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_chain(Avbd2dBodyId body, const Avbd2dChainDef *def, Avbd2dChainId *out_id)
    {
        BODY_CTX(body);
        NEED(def);
        NEED(out_id);
        if (!def->points || def->count < (def->isLoop ? 3 : 2))
            return AVBD2D_ERR_INVALID_ARG;
        for (int i = 0; i < def->count; i++)
            if (!finite2(def->points[i]))
                return AVBD2D_ERR_INVALID_ARG;
        const int count = def->count;
        const int n = def->isLoop ? count : count - 1;
        const Avbd2dVec2 *pts = def->points;
        auto P = [&](int i) -> Vec2
        {
            if (def->isLoop)
                return V(pts[((i % count) + count) % count]);
            if (i < 0)
                return V(pts[0]) * 2.0f - V(pts[1]);
            if (i >= count)
                return V(pts[count - 1]) * 2.0f - V(pts[count - 2]);
            return V(pts[i]);
        };
        return guarded(w, true,
                       [&]() -> Avbd2dResult
                       {
                           Avbd2dShapeDef sd = avbd2d_default_shape_def();
                           sd.friction = def->friction;
                           sd.restitution = def->restitution;
                           sd.filter = def->filter;
                           sd.userData = def->userData;
                           sd.density = 0.0f;
                           int first = -1;
                           for (int i = 0; i < n; i++)
                           {
                               const int s = w->shapes.acquire();
                               ShapeRec &r = w->shapes.items[(size_t)s];
                               r.body = slot;
                               r.kind = SK_CHAIN;
                               r.chain[0] = P(i - 1);
                               r.chain[1] = P(i);
                               r.chain[2] = P(i + 1);
                               r.chain[3] = P(i + 2);
                               r.def = sd;
                               b->shapes.push_back(s);
                               if (first < 0)
                                   first = s;
                           }
                           markReshape(w, slot);
                           *out_id = Avbd2dChainId{first + 1, w->id, w->shapes.items[(size_t)first].gen};
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy_shape(Avbd2dShapeId id)
    {
        SHAPE_CTX(id);
        const int bodySlot = s->body;
        BodyRec &b = w->bodies.items[(size_t)bodySlot];
        b.shapes.erase(std::remove(b.shapes.begin(), b.shapes.end(), slot), b.shapes.end());
        retireShape(w, slot);
        markReshape(w, bodySlot);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy_chain(Avbd2dChainId id)
    {
        ID_WORLD(id);
        if (!avbd2d_chain_is_valid(id))
            return AVBD2D_ERR_INVALID_ARG;
        // A chain is its run of consecutive chain-segment shapes on one body, starting at `id`.
        const int first = id.index1 - 1;
        const int bodySlot = w->shapes.items[(size_t)first].body;
        BodyRec &b = w->bodies.items[(size_t)bodySlot];
        std::vector<int> run;
        size_t at = 0;
        while (at < b.shapes.size() && b.shapes[at] != first)
            at++;
        // The segments of one chain were created together, so they are adjacent in the body's list.
        for (size_t k = at; k < b.shapes.size() && w->shapes.items[(size_t)b.shapes[k]].kind == SK_CHAIN; k++)
            run.push_back(b.shapes[k]);
        for (int s : run)
        {
            b.shapes.erase(std::remove(b.shapes.begin(), b.shapes.end(), s), b.shapes.end());
            retireShape(w, s);
        }
        markReshape(w, bodySlot);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_shape_get_body(Avbd2dShapeId id, Avbd2dBodyId *out)
    {
        SHAPE_CTX(id);
        NEED(out);
        *out = bodyIdOf(w, s->body);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_shape_get_user_data(Avbd2dShapeId id, void **out)
    {
        SHAPE_CTX(id);
        NEED(out);
        *out = s->def.userData;
        return AVBD2D_OK;
    }
} // extern "C"
