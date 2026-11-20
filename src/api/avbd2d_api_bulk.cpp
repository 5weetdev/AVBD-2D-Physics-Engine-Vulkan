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


// Bulk body and joint operations of the 2D C ABI.

#include "avbd2d_world.h"

template <class T> static const T &strided(const T *base, int32_t stride, int64_t i)
{
    return *reinterpret_cast<const T *>(reinterpret_cast<const char *>(base) + (int64_t)stride * i);
}

extern "C"
{
    // ----- Bulk create and destroy -----------------------------------------------------------------
    // Element i of a strided array (stride in bytes; 0 repeats element 0).

    static bool validBodyDef(const Avbd2dBodyDef &d)
    {
        return finite2(d.position) && finiteRot(d.rotation) && finite2(d.linearVelocity) &&
               finite1(d.angularVelocity) && d.type >= AVBD2D_STATIC_BODY && d.type <= AVBD2D_DYNAMIC_BODY;
    }

    // The shape record of a geometry, or false when it is not a valid shape (the same rules as the
    // single create calls).
    static bool geometryToRec(const Avbd2dShapeGeometry &g, ShapeRec &r)
    {
        switch (g.type)
        {
        case AVBD2D_SHAPE_CIRCLE:
            if (!(g.radius > 0.0f) || !finite1(g.radius) || !finite2(g.points[0]))
                return false;
            r.kind = SK_CIRCLE;
            r.circle = Avbd2dCircle{g.points[0], g.radius};
            return true;
        case AVBD2D_SHAPE_CAPSULE:
            if (!(g.radius > 0.0f) || !finite1(g.radius) || !finite2(g.points[0]) || !finite2(g.points[1]))
                return false;
            r.kind = SK_CAPSULE;
            r.capsule = Avbd2dCapsule{g.points[0], g.points[1], g.radius};
            return true;
        case AVBD2D_SHAPE_SEGMENT:
            if (!finite2(g.points[0]) || !finite2(g.points[1]))
                return false;
            r.kind = SK_SEGMENT;
            r.seg = Avbd2dSegment{g.points[0], g.points[1]};
            return true;
        case AVBD2D_SHAPE_POLYGON:
        {
            if (g.count < 3 || g.count > AVBD2D_MAX_POLYGON_VERTICES || !(g.radius >= 0.0f) || !finite1(g.radius))
                return false;
            Vec2 in[AVBD2D_MAX_POLYGON_VERTICES], hull[AVBD2D_MAX_POLYGON_VERTICES];
            for (int i = 0; i < g.count; i++)
            {
                if (!finite2(g.points[i]))
                    return false;
                in[i] = V(g.points[i]);
            }
            const int n = Scene2D::computeHull(in, g.count, hull);
            if (n < 3)
                return false;
            r.kind = SK_POLYGON;
            r.poly = fillPolygon(hull, n, g.radius);
            return true;
        }
        case AVBD2D_SHAPE_BOX:
        {
            const Avbd2dVec2 h = g.points[0], c = g.points[1];
            Avbd2dRot q = Avbd2dRot{g.points[2].x, g.points[2].y};
            if (!(h.x > 0.0f) || !(h.y > 0.0f) || !finite2(h) || !finite2(c) || !finiteRot(q))
                return false;
            if (q.c == 0.0f && q.s == 0.0f)
                q = Avbd2dRot{1.0f, 0.0f};
            r.kind = SK_POLYGON;
            r.poly = avbd2d_make_offset_box(h.x, h.y, c, q);
            return true;
        }
        default:
            return false;
        }
    }

    AVBD2D_API Avbd2dResult avbd2d_create_bodies(Avbd2dWorld *world, const Avbd2dBodyBatch *batch,
                                                 Avbd2dBodyId *out_bodies, Avbd2dShapeId *out_shapes)
    {
        AVBD2D_GET_WORLD(world);
        NEED(batch);
        NEED(batch->struct_size >= sizeof(Avbd2dBodyBatch));
        const int32_t count = batch->count;
        NEED(count >= 0);
        if (count == 0)
            return AVBD2D_OK;
        NEED(batch->bodyDefs);
        NEED(out_bodies);

        // 1. Validate everything; nothing changes on an error.
        int64_t totalShapes = 0;
        for (int32_t i = 0; i < count; i++)
        {
            Avbd2dBodyDef d = strided(batch->bodyDefs, batch->bodyDefStride, i);
            if (batch->transforms)
            {
                const Avbd2dTransform &t = strided(batch->transforms, batch->transformStride, i);
                d.position = t.p;
                d.rotation = t.q;
            }
            if (batch->velocities)
            {
                const Avbd2dBodyVelocity &v = strided(batch->velocities, batch->velocityStride, i);
                d.linearVelocity = v.linear;
                d.angularVelocity = v.angular;
            }
            NEED(validBodyDef(d));
            if (batch->shapeCounts)
            {
                const int32_t n = strided(batch->shapeCounts, batch->shapeCountStride, i);
                NEED(n >= 0);
                totalShapes += n;
            }
        }
        NEED(totalShapes <= (int64_t)INT32_MAX);
        if (totalShapes > 0)
        {
            NEED(batch->geometries);
            NEED(batch->shapeDefs);
            ShapeRec scratch;
            for (int64_t k = 0; k < totalShapes; k++)
            {
                NEED(geometryToRec(strided(batch->geometries, batch->geometryStride, k), scratch));
                NEED(validShapeDef(&strided(batch->shapeDefs, batch->shapeDefStride, k)));
            }
        }

        // 2. Create. The tables are grown first, so the loop itself does not reallocate them.
        return guarded(w, true,
                       [&]() -> Avbd2dResult
                       {
                           w->bodies.items.reserve(w->bodies.items.size() + (size_t)count);
                           w->shapes.items.reserve(w->shapes.items.size() + (size_t)totalShapes);
                           w->pendingBodies.reserve(w->pendingBodies.size() + (size_t)count);
                           w->viewDirty.reserve(w->viewDirty.size() + (size_t)count);
                           int64_t k = 0;
                           for (int32_t i = 0; i < count; i++)
                           {
                               const int slot = w->bodies.acquire();
                               BodyRec &b = w->bodies.items[(size_t)slot];
                               b.def = strided(batch->bodyDefs, batch->bodyDefStride, i);
                               if (batch->transforms)
                               {
                                   const Avbd2dTransform &t = strided(batch->transforms, batch->transformStride, i);
                                   b.def.position = t.p;
                                   b.def.rotation = t.q;
                               }
                               if (batch->velocities)
                               {
                                   const Avbd2dBodyVelocity &v = strided(batch->velocities, batch->velocityStride, i);
                                   b.def.linearVelocity = v.linear;
                                   b.def.angularVelocity = v.angular;
                               }
                               if (batch->userData)
                                   b.def.userData = strided(batch->userData, batch->userDataStride, i);
                               const int32_t ns =
                                   batch->shapeCounts ? strided(batch->shapeCounts, batch->shapeCountStride, i) : 0;
                               b.shapes.reserve((size_t)ns);
                               for (int32_t j = 0; j < ns; j++, k++)
                               {
                                   const int s = w->shapes.acquire();
                                   ShapeRec &r = w->shapes.items[(size_t)s];
                                   const uint16_t gen = r.gen;
                                   r = ShapeRec();
                                   geometryToRec(strided(batch->geometries, batch->geometryStride, k), r);
                                   r.alive = true;
                                   r.gen = gen;
                                   r.body = slot;
                                   r.def = strided(batch->shapeDefs, batch->shapeDefStride, k);
                                   b.shapes.push_back(s);
                                   if (out_shapes)
                                       out_shapes[k] = shapeIdOf(w, s);
                               }
                               w->pendingBodies.push_back(slot);
                               markDirty(w, slot);
                               out_bodies[i] = bodyIdOf(w, slot);
                           }
                           return AVBD2D_OK;
                       });
    }

    // Every id valid, of this world, and listed once; `mark` is left all zero.
    static bool validBodyIds(Avbd2dWorld *w, const Avbd2dBodyId *ids, int32_t count, std::vector<uint8_t> *mark)
    {
        if (count < 0 || (!ids && count > 0))
            return false;
        if (mark)
            mark->assign(w->bodies.items.size(), 0);
        bool ok = true;
        int32_t i = 0;
        for (; i < count; i++)
        {
            const Avbd2dBodyId id = ids[i];
            if (id.world0 != w->id || !w->bodies.find(id.index1, id.generation))
            {
                ok = false;
                break;
            }
            if (mark)
            {
                uint8_t &m = (*mark)[(size_t)id.index1 - 1];
                if (m)
                {
                    ok = false;
                    break;
                }
                m = 1;
            }
        }
        if (mark && !ok)
            for (int32_t j = 0; j < i; j++)
                (*mark)[(size_t)ids[j].index1 - 1] = 0;
        return ok;
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy_bodies(Avbd2dWorld *world, const Avbd2dBodyId *ids, int32_t count)
    {
        AVBD2D_GET_WORLD(world);
        std::vector<uint8_t> mark;
        if (!validBodyIds(w, ids, count, &mark))
            return AVBD2D_ERR_INVALID_ARG;
        return guarded(
            w, false,
            [&]() -> Avbd2dResult
            {
                for (int32_t i = 0; i < count; i++)
                    destroyBodySlot(w, ids[i].index1 - 1, false);
                // One pass over each list for the whole batch.
                auto marked = [&](int slot) { return (size_t)slot < mark.size() && mark[(size_t)slot]; };
                w->pendingBodies.erase(std::remove_if(w->pendingBodies.begin(), w->pendingBodies.end(), marked),
                                       w->pendingBodies.end());
                w->replaceBodies.erase(std::remove_if(w->replaceBodies.begin(), w->replaceBodies.end(), marked),
                                       w->replaceBodies.end());
                return AVBD2D_OK;
            });
    }

// ----- Bulk setters ----------------------------------------------------------------------------
#define BULK_CTX(world, ids, count)                                                                                    \
    AVBD2D_GET_WORLD(world);                                                                                           \
    if (!validBodyIds(w, ids, count, nullptr))                                                                         \
    return AVBD2D_ERR_INVALID_ARG

    static BodyRec &recOf(Avbd2dWorld *w, Avbd2dBodyId id)
    {
        return w->bodies.items[(size_t)id.index1 - 1];
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_set_transforms(Avbd2dWorld *world, const Avbd2dBodyId *ids,
                                                         const Avbd2dTransform *transforms, int32_t count)
    {
        BULK_CTX(world, ids, count);
        NEED(transforms || count == 0);
        for (int32_t i = 0; i < count; i++)
            NEED(finite2(transforms[i].p) && finiteRot(transforms[i].q));
        for (int32_t i = 0; i < count; i++)
            setTransformOn(w, recOf(w, ids[i]), ids[i].index1 - 1, transforms[i].p, transforms[i].q);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_set_velocities(Avbd2dWorld *world, const Avbd2dBodyId *ids,
                                                         const Avbd2dBodyVelocity *velocities, int32_t count)
    {
        BULK_CTX(world, ids, count);
        NEED(velocities || count == 0);
        for (int32_t i = 0; i < count; i++)
            NEED(finite2(velocities[i].linear) && finite1(velocities[i].angular));
        for (int32_t i = 0; i < count; i++)
            setVelocityOn(w, recOf(w, ids[i]), ids[i].index1 - 1, &velocities[i].linear, &velocities[i].angular);
        return AVBD2D_OK;
    }

    // Impulses (scale 1) or forces (scale dt).
    static Avbd2dResult applyWrenches(Avbd2dWorld *w, const Avbd2dBodyId *ids, const Avbd2dBodyWrench *wr,
                                      int32_t count, float scale)
    {
        NEED(wr || count == 0);
        for (int32_t i = 0; i < count; i++)
            NEED(finite2(wr[i].linear) && finite1(wr[i].angular) &&
                 (!(wr[i].flags & AVBD2D_WRENCH_AT_POINT) || finite2(wr[i].point)));
        for (int32_t i = 0; i < count; i++)
        {
            const Vec2 p = V(wr[i].point);
            const bool atPoint = (wr[i].flags & AVBD2D_WRENCH_AT_POINT) != 0;
            impulseOn(w, recOf(w, ids[i]), V(wr[i].linear) * scale, atPoint ? &p : nullptr, wr[i].angular * scale,
                      (wr[i].flags & AVBD2D_WRENCH_WAKE) ? 1 : 0);
        }
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_apply_impulses(Avbd2dWorld *world, const Avbd2dBodyId *ids,
                                                         const Avbd2dBodyWrench *impulses, int32_t count)
    {
        BULK_CTX(world, ids, count);
        return applyWrenches(w, ids, impulses, count, 1.0f);
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_apply_forces(Avbd2dWorld *world, const Avbd2dBodyId *ids,
                                                       const Avbd2dBodyWrench *forces, int32_t count)
    {
        BULK_CTX(world, ids, count);
        return applyWrenches(w, ids, forces, count, w->world->params().dt);
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_set_awake(Avbd2dWorld *world, const Avbd2dBodyId *ids, int32_t count,
                                                    int32_t awake)
    {
        BULK_CTX(world, ids, count);
        for (int32_t i = 0; i < count; i++)
            setAwakeOn(w, recOf(w, ids[i]), ids[i].index1 - 1, awake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_bodies_set_enabled(Avbd2dWorld *world, const Avbd2dBodyId *ids, int32_t count,
                                                      int32_t enabled)
    {
        BULK_CTX(world, ids, count);
        for (int32_t i = 0; i < count; i++)
            setEnabledOn(w, recOf(w, ids[i]), ids[i].index1 - 1, enabled);
        return AVBD2D_OK;
    }

    // ----- Bulk joint setters ----------------------------------------------------------------------
    // Every id a live revolute joint of this world, listed once; `mark` is left all zero.
    static bool validJointIds(Avbd2dWorld *w, const Avbd2dJointId *ids, int32_t count, std::vector<uint8_t> *mark)
    {
        if (count < 0 || (!ids && count > 0))
            return false;
        if (mark)
            mark->assign(w->joints.items.size(), 0);
        bool ok = true;
        int32_t i = 0;
        for (; i < count; i++)
        {
            const Avbd2dJointId id = ids[i];
            JointRec *j = id.world0 == w->id ? w->joints.find(id.index1, id.generation) : nullptr;
            if (!j || j->d.type != JointType::Revolute)
            {
                ok = false;
                break;
            }
            if (mark)
            {
                uint8_t &m = (*mark)[(size_t)id.index1 - 1];
                if (m)
                {
                    ok = false;
                    break;
                }
                m = 1;
            }
        }
        if (mark && !ok)
            for (int32_t k = 0; k < i; k++)
                (*mark)[(size_t)ids[k].index1 - 1] = 0;
        return ok;
    }

    AVBD2D_API Avbd2dResult avbd2d_revolute_joints_set_drives(Avbd2dWorld *world, const Avbd2dJointId *ids,
                                                              const Avbd2dRevoluteDrive *drives, int32_t count)
    {
        AVBD2D_GET_WORLD(world);
        NEED(drives || count == 0);
        std::vector<uint8_t> mark;
        if (!validJointIds(w, ids, count, &mark))
            return AVBD2D_ERR_INVALID_ARG;
        for (int32_t i = 0; i < count; i++)
        {
            const Avbd2dRevoluteDrive &d = drives[i];
            NEED(d.mode >= AVBD2D_DRIVE_OFF && d.mode <= AVBD2D_DRIVE_TARGET);
            NEED(finite1(d.target) && finite1(d.maxTorque) && d.maxTorque >= 0.0f);
        }
        for (int32_t i = 0; i < count; i++)
        {
            JointRec *j = w->joints.find(ids[i].index1, ids[i].generation); // validJointIds proved it
            j->d.enableMotor = drives[i].mode != AVBD2D_DRIVE_OFF;
            j->d.motorMode = (uint32_t)drives[i].mode;
            j->d.motorTarget = drives[i].target;
            j->d.maxMotorTorque = drives[i].maxTorque;
            syncMotor(w, *j);
        }
        return AVBD2D_OK;
    }
} // extern "C"
