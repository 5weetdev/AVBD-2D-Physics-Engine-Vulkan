// avbd2d_demo: the 2D AVBD solver (VkWorld2D) with a flat-colour instanced
// renderer, SDL2 + Dear ImGui on Vulkan.
//
//   left mouse   grab a body        right/middle mouse   pan        wheel   zoom
//   Space pause   . single step     R reset the scene    Esc quit
//   F            spawn one object (box, disc or ragdoll, per the Spawn panel) at the cursor
//   G            spawn a bunch of them                   W   wake everything
//   P            capture the scene as it is now to captures2d/*.a2dcap; replay it from the
//                Captures panel or with --scene <file.a2dcap> (R restarts the capture)
//
// The solver and the renderer share one Vulkan device: VkWorld2D owns it, and this file puts
// the swapchain on it. Bodies are drawn from the pose copy the step makes inside its own
// submission, so nothing here waits on the GPU beyond the frame fence.

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <vulkan/vulkan.h>

#include "renderer2d.h"
#include "scene_io2d.h"
#include "scenes2d.h"
#include "vk_util.h"
#include "vk_world2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>


using namespace avbd2d;

static constexpr int kFramesInFlight = 2;
static constexpr int kWinWidth = 1280;
static constexpr int kWinHeight = 720;

static SDL_Window *g_window = nullptr;
static VkWorld2D g_world;
static Renderer2D g_renderer;
static avbdvk::Device *g_dev = nullptr;

static VkSurfaceKHR g_surface = VK_NULL_HANDLE;
static VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
static VkFormat g_swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
static VkExtent2D g_swapExtent{kWinWidth, kWinHeight};
static std::vector<VkImage> g_swapImages;
static std::vector<VkImageView> g_swapViews;
static std::vector<VkFramebuffer> g_framebuffers;
// One render-finished semaphore per swapchain image: the present waits on the acquired
// image's, which does not cycle in step with the frame-in-flight index.
static std::vector<VkSemaphore> g_renderDone;
static VkRenderPass g_renderPass = VK_NULL_HANDLE;
static VkCommandPool g_cmdPool = VK_NULL_HANDLE;
static VkCommandBuffer g_cmdBufs[kFramesInFlight]{};
static VkSemaphore g_imageAvail[kFramesInFlight]{};
static VkFence g_inFlight[kFramesInFlight]{};
static int g_frameIdx = 0;
static bool g_vsync = true;
static bool g_swapDirty = false;

// --- Scene / camera state ---------------------------------------------------------------
static int g_sceneIndex = 0;
static int g_sceneCount = 0;
static const scenes::SceneEntry2D *g_scenes = nullptr;
static std::vector<uint32_t> g_colors;
static Vec2 g_camCenter{0.0f, 5.0f};
static float g_camHeight = 20.0f;
static bool g_camKept = false; // the first scene load places the camera, later ones keep the user's
static bool g_paused = false;
static bool g_stepOnce = false;
static bool g_panning = false;
static int g_stepFailures = 0;
static int g_nonFinite = 0;
static StepResult2D g_lastResult;
static const char *g_warning = nullptr; // why the simulation paused itself, if it did
static Vec2 g_cursor{0.0f, 0.0f};       // mouse position in world space

// Spawning.
static int g_spawnKind = 0; // 0 box, 1 disc, 2 ragdoll, 3 capsule, 4 polygon, 5 A-pose ragdoll
static float g_ragdollBreak = 150.0f;
static float g_spawnSize = 0.6f;
static float g_spawnDensity = 1.0f;
static float g_spawnFriction = 0.5f;
static int g_spawnCount = 100;
static bool g_rain = false;
static float g_rainRate = 200.0f; // bodies per second
static float g_rainCarry = 0.0f;
static uint32_t g_spawnSeed = 1u;

// Grab (the Grab panel). The radius applies when "grab connected bodies" is on.
static bool g_grabConnected = true;
static bool g_grabFollowJoints = true;
static float g_grabRadius = 3.0f;
static float g_grabFalloff = 2.0f;
static float g_grabMinStrength = 0.05f;
static bool g_paramsKept = false; // the first scene load takes the scene's params, later ones keep the panel's

static VkWorld2D::GrabOptions grabOptions()
{
    VkWorld2D::GrabOptions o;
    o.radius = g_grabConnected ? g_grabRadius : 0.0f;
    o.followJoints = g_grabFollowJoints;
    o.falloff = g_grabFalloff;
    o.minStrength = g_grabMinStrength;
    return o;
}

// Overlays.
static int g_colorMode = 0; // 0 body colour, 1 sleep state, 2 speed, 3 graph colour
struct EventMark
{
    Vec2 p;
    int life;
    int kind; // 0 contact begin, 1 hit, 2 sensor begin, 3 sensor end
};
static bool g_showEvents = false;
static std::vector<EventMark> g_eventMarks;
static bool g_showContacts = false;
static bool g_showNormals = false;
static bool g_showJoints = false;
static bool g_showGrid = false;

// Step-time history for the plot.
static constexpr int kHistory = 180;
static float g_histHost[kHistory];
static float g_histGpu[kHistory];
static int g_histPos = 0;

// ---------------------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------------------
static VkSurfaceFormatKHR chooseSurfaceFormat()
{
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_dev->physicalDevice(), g_surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_dev->physicalDevice(), g_surface, &n, formats.data());
    for (auto &f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats[0];
}

static void destroySwapchainObjects()
{
    for (auto fb : g_framebuffers)
        if (fb)
            vkDestroyFramebuffer(g_dev->device(), fb, nullptr);
    g_framebuffers.clear();
    for (auto v : g_swapViews)
        if (v)
            vkDestroyImageView(g_dev->device(), v, nullptr);
    g_swapViews.clear();
    for (auto s : g_renderDone)
        if (s)
            vkDestroySemaphore(g_dev->device(), s, nullptr);
    g_renderDone.clear();
}

static bool createSwapchain()
{
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_dev->physicalDevice(), g_surface, &caps);

    int w = 0, h = 0;
    SDL_Vulkan_GetDrawableSize(g_window, &w, &h);
    if (w == 0 || h == 0)
        return false; // minimised

    g_swapExtent.width = std::min(std::max((uint32_t)w, caps.minImageExtent.width), caps.maxImageExtent.width);
    g_swapExtent.height = std::min(std::max((uint32_t)h, caps.minImageExtent.height), caps.maxImageExtent.height);

    const VkSurfaceFormatKHR fmt = chooseSurfaceFormat();
    g_swapFormat = fmt.format;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    bool mailbox = false;
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_dev->physicalDevice(), g_surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_dev->physicalDevice(), g_surface, &n, modes.data());
        for (auto pm : modes)
            if (pm == VK_PRESENT_MODE_MAILBOX_KHR)
                mailbox = true;
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = g_surface;
    sci.minImageCount = imageCount;
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = g_swapExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = (g_vsync || !mailbox) ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = g_swapchain;
    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(g_dev->device(), &sci, nullptr, &newSwapchain));

    if (g_swapchain)
    {
        destroySwapchainObjects();
        vkDestroySwapchainKHR(g_dev->device(), g_swapchain, nullptr);
    }
    g_swapchain = newSwapchain;

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(g_dev->device(), g_swapchain, &n, nullptr);
    g_swapImages.resize(n);
    vkGetSwapchainImagesKHR(g_dev->device(), g_swapchain, &n, g_swapImages.data());

    g_swapViews.resize(n);
    g_framebuffers.resize(n);
    g_renderDone.resize(n);
    for (uint32_t i = 0; i < n; i++)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = g_swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = g_swapFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(g_dev->device(), &vci, nullptr, &g_swapViews[i]));

        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = g_renderPass;
        fci.attachmentCount = 1;
        fci.pAttachments = &g_swapViews[i];
        fci.width = g_swapExtent.width;
        fci.height = g_swapExtent.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(g_dev->device(), &fci, nullptr, &g_framebuffers[i]));

        VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(g_dev->device(), &semi, nullptr, &g_renderDone[i]));
    }
    return true;
}

static void createRenderPass()
{
    VkAttachmentDescription att{};
    att.format = g_swapFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(g_dev->device(), &rpci, nullptr, &g_renderPass));
}

// ---------------------------------------------------------------------------------------
// Colours, scenes, camera
// ---------------------------------------------------------------------------------------
static uint32_t packRGB(float r, float g, float b)
{
    auto q = [](float v) { return (uint32_t)(std::min(std::max(v, 0.0f), 1.0f) * 255.0f + 0.5f); };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (255u << 24);
}

static uint32_t hsv(float h, float s, float v)
{
    h = h - std::floor(h);
    const float c = v * s;
    const float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    const float m = v - c;
    float r = 0, g = 0, b = 0;
    switch ((int)(h * 6.0f) % 6)
    {
    case 0: r = c; g = x; break;
    case 1: r = x; g = c; break;
    case 2: g = c; b = x; break;
    case 3: g = x; b = c; break;
    case 4: r = x; b = c; break;
    default: r = c; b = x; break;
    }
    return packRGB(r + m, g + m, b + m);
}

static const uint32_t kStaticColor = packRGB(0.34f, 0.37f, 0.44f);

// Fills g_colors for every body that does not have one yet.
static void extendColors()
{
    const Scene2D &s = g_world.scene();
    const size_t old = g_colors.size();
    g_colors.resize((size_t)s.bodyCount(), 0u);
    for (size_t i = old; i < g_colors.size(); i++)
    {
        if (s.color[i] != 0)
            g_colors[i] = s.color[i];
        else if (s.mass[i] <= 0.0f)
            g_colors[i] = kStaticColor;
        else
            g_colors[i] = hsv(0.61803398875f * (float)i, s.shType[(size_t)s.shapeFirst[(size_t)i]] == SHAPE2D_CIRCLE ? 0.45f : 0.6f, 0.92f);
    }
}

// Scene captures (key P): the running world frozen into a file under captures2d/, replayed as a
// scene. While one is loaded R and Reset restart the capture, with the capture's own solver
// parameters so a benchmark is the same on every run.
static constexpr const char *kCaptureDir = "captures2d";
static bool g_inCapture = false;
static Scene2D g_captureScene;
static CaptureMeta2D g_captureMeta;
static std::string g_captureName;     // file name of the loaded capture
static std::string g_captureStatus;   // last capture / load message for the panel
static std::vector<std::string> g_captureFiles;
static uint64_t g_stepsTaken = 0;     // steps since the scene was loaded

static void refreshCaptureList()
{
    g_captureFiles.clear();
    std::error_code ec;
    for (const auto &de : std::filesystem::directory_iterator(kCaptureDir, ec))
        if (de.is_regular_file(ec) && de.path().extension() == kCaptureExt2D)
            g_captureFiles.push_back(de.path().filename().string());
    std::sort(g_captureFiles.begin(), g_captureFiles.end());
}

// Builds `s` into the world and resets the demo state around it.
static void installScene(const Scene2D &s, bool keepPanelParams, bool events)
{
    // The panel's simulation, sleep and kill settings are the user's: they survive a scene change.
    const SolverParams2D keep = g_world.params();
    g_world.build(s, keepPanelParams && g_paramsKept ? &keep : nullptr);
    g_paramsKept = true;
    g_stepsTaken = 0;
    g_showEvents = events;
    g_world.setEventsEnabled(g_showEvents);
    g_eventMarks.clear();
    g_colors.clear();
    extendColors();
    // The camera is the user's too: a reload (R / Reset / scene combo / capture) keeps the view.
    if (!g_camKept)
    {
        g_camCenter = s.cameraCenter;
        g_camHeight = s.cameraHeight;
        g_camKept = true;
    }
    g_paused = false;
    g_warning = nullptr;
    g_lastResult = StepResult2D();
    g_rainCarry = 0.0f;
    memset(g_histHost, 0, sizeof(g_histHost));
    memset(g_histGpu, 0, sizeof(g_histGpu));
}

static void loadScene(int index)
{
    g_sceneIndex = (index % g_sceneCount + g_sceneCount) % g_sceneCount;
    g_inCapture = false;
    Scene2D s;
    g_scenes[g_sceneIndex].build(s);
    installScene(s, true, strcmp(g_scenes[g_sceneIndex].name, "Sensor Funnel") == 0);
}

// Replays the capture in g_captureScene.
static void replayCapture()
{
    g_inCapture = true;
    installScene(g_captureScene, false, false);
    g_paramsKept = false; // the next catalogue scene takes its own parameters, not the capture's
}

static bool loadCapture(const std::string &path)
{
    Scene2D s;
    CaptureMeta2D meta;
    std::string err, warn;
    if (!loadScene2D(s, path.c_str(), &meta, &err, &warn))
    {
        g_captureStatus = err;
        fprintf(stderr, "avbd2d_demo: %s\n", err.c_str());
        return false;
    }
    g_captureScene = std::move(s);
    g_captureMeta = meta;
    g_captureName = std::filesystem::path(path).filename().string();
    replayCapture();
    g_captureStatus = "loaded " + g_captureName + (warn.empty() ? "" : " (" + warn + ")");
    return true;
}

// Key P: freezes the world as it is now into captures2d/<scene>_<time>.a2dcap.
static void captureNow()
{
    const int n = g_world.bodyCount();
    std::vector<uint8_t> removed, broken;
    g_world.downloadLiveFlags(removed, broken);
    Scene2D cap = captureScene2D(g_world.scene(), g_world.positions(), g_world.angles(), g_world.velocities(),
                                 g_world.angularVelocities(), removed, broken);
    CaptureMeta2D meta;
    meta.source = g_inCapture ? g_captureMeta.source : g_scenes[g_sceneIndex].name;
    meta.step = g_stepsTaken + (g_inCapture ? g_captureMeta.step : 0);
    // The camera as the user has it, so starting on this capture (--scene file.a2dcap)
    // opens on the same view.
    cap.cameraCenter = g_camCenter;
    cap.cameraHeight = g_camHeight;
    cap.params = g_world.params();

    std::string stem;
    for (const char c : meta.source)
        stem += (isalnum((unsigned char)c) ? c : '_');
    char stamp[32];
    const time_t now = time(nullptr);
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", localtime(&now));
    std::error_code ec;
    std::filesystem::create_directories(kCaptureDir, ec);
    const std::string path = std::string(kCaptureDir) + "/" + stem + "_" + stamp + kCaptureExt2D;
    std::string err;
    if (saveScene2D(cap, path.c_str(), meta, &err))
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "captured %d of %d bodies, %d joints -> %s", cap.bodyCount(), n, cap.jointCount(),
                 path.c_str());
        g_captureStatus = msg;
        printf("avbd2d_demo: %s\n", msg);
    }
    else
    {
        g_captureStatus = err;
        fprintf(stderr, "avbd2d_demo: %s\n", err.c_str());
    }
    refreshCaptureList();
}

// R / Reset: restart whatever is loaded.
static void restartScene()
{
    if (g_inCapture)
        replayCapture();
    else
        loadScene(g_sceneIndex);
}

struct View
{
    float x0, x1, y0, y1;
};

static float drawableAspect()
{
    int dw = 1, dh = 1;
    SDL_Vulkan_GetDrawableSize(g_window, &dw, &dh);
    return dh > 0 ? (float)dw / (float)dh : 1.0f;
}

static View viewRect()
{
    const float w = g_camHeight * drawableAspect();
    return {g_camCenter.x - 0.5f * w, g_camCenter.x + 0.5f * w, g_camCenter.y - 0.5f * g_camHeight,
            g_camCenter.y + 0.5f * g_camHeight};
}

static void camera(float cam[4])
{
    cam[0] = g_camCenter.x;
    cam[1] = g_camCenter.y;
    cam[2] = 2.0f / (g_camHeight * drawableAspect());
    cam[3] = 2.0f / g_camHeight;
}

// Window pixel -> world point (window coordinates, so HiDPI scaling does not matter).
static Vec2 screenToWorld(int mx, int my)
{
    int ww = 1, wh = 1;
    SDL_GetWindowSize(g_window, &ww, &wh);
    const float u = (float)mx / (float)std::max(ww, 1);
    const float v = (float)my / (float)std::max(wh, 1);
    return Vec2(g_camCenter.x + (u - 0.5f) * g_camHeight * drawableAspect(), g_camCenter.y - (v - 0.5f) * g_camHeight);
}

// ---------------------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------------------
static float rnd()
{
    g_spawnSeed = g_spawnSeed * 1664525u + 1013904223u;
    return (float)((g_spawnSeed >> 8) & 0xFFFF) / 65535.0f;
}

static uint32_t randomShirt()
{
    return scenes::rgb8(0.35f + 0.65f * rnd(), 0.3f + 0.5f * rnd(), 0.3f + 0.6f * rnd());
}

// Shape and jittered size of one spawned object of demo kind `kind` (not a ragdoll).
static void setSpawnShape(SpawnDesc2D &o, int kind)
{
    const float j = 0.8f + 0.4f * rnd();
    const float sz = g_spawnSize * j;
    switch (kind)
    {
    case 1:
        o.shape = SHAPE2D_CIRCLE;
        o.size = Vec2(0.5f * sz, 0.0f);
        break;
    case 3:
        o.shape = SHAPE2D_CAPSULE;
        o.size = Vec2(sz * (1.4f + 0.8f * rnd()), 0.5f * sz);
        break;
    case 4:
        o.shape = SHAPE2D_POLYGON;
        o.size = Vec2(0.55f * sz, 0.0f);
        o.sides = 3 + (int)(rnd() * 5.99f);
        o.radius = rnd() < 0.3f ? 0.08f * sz : 0.0f;
        break;
    default:
        o.shape = SHAPE2D_BOX;
        o.size = Vec2(sz, g_spawnSize * (0.8f + 0.4f * rnd()));
        break;
    }
}

// `count` ragdolls in a loose grid whose bottom row stands at `center`.
static void spawnRagdolls(Vec2 center, int count, bool aPose = false)
{
    count = std::min(count, 2000);
    if (count <= 0)
        return;
    Scene2D frag;
    const float scale = std::max(g_spawnSize / 0.43f, 0.3f);
    const int cols = std::max(1, (int)std::ceil(std::sqrt((double)count)));
    const float dx = (aPose ? 1.5f : 1.0f) * scale, dy = 1.9f * scale;
    for (int k = 0; k < count; k++)
    {
        const int gx = k % cols, gy = k / cols;
        const Vec2 at(((float)gx - 0.5f * (float)(cols - 1)) * dx, (float)gy * dy);
        const float brk = g_ragdollBreak > 0.0f ? g_ragdollBreak : kStiffInf;
        if (aPose)
            scenes::addPoseRagdoll(frag, at, scale, brk, randomShirt());
        else
            scenes::addRagdoll(frag, at, scale, brk, randomShirt());
    }
    g_world.spawnScene(frag, Vec2(center.x, center.y - 0.8f * scale));
    extendColors();
}

// `count` objects of the selected kind in a loose grid centred on `center`, all moving at `vel`.
static void spawnCluster(Vec2 center, int kind, int count, Vec2 vel)
{
    if (count <= 0)
        return;
    if (kind == 2 || kind == 5)
    {
        spawnRagdolls(center, count, kind == 5);
        return;
    }
    count = std::min(count, 200000);
    std::vector<SpawnDesc2D> d((size_t)count);
    const int cols = std::max(1, (int)std::ceil(std::sqrt((double)count)));
    const float spacing = g_spawnSize * (kind == 3 ? 2.8f : 1.25f);
    for (int k = 0; k < count; k++)
    {
        SpawnDesc2D &o = d[(size_t)k];
        const int gx = k % cols, gy = k / cols;
        setSpawnShape(o, kind);
        o.pos = Vec2(center.x + ((float)gx - 0.5f * (float)(cols - 1)) * spacing, center.y + (float)gy * spacing);
        o.angle = count == 1 ? 0.0f : kPi * rnd();
        o.density = g_spawnDensity;
        o.friction = g_spawnFriction;
        o.vel = vel;
    }
    g_world.spawn(d.data(), count);
    extendColors();
}

// Bodies falling from just above the top of the view, spread across its width.
static void spawnRain(int count)
{
    if (count <= 0)
        return;
    const View v = viewRect();
    if (g_spawnKind == 2 || g_spawnKind == 5)
    {
        Scene2D frag;
        const float scale = std::max(g_spawnSize / 0.43f, 0.3f);
        for (int k = 0; k < std::min(count, 50); k++)
        {
            const Vec2 at(v.x0 + (v.x1 - v.x0) * rnd(), v.y1 + 1.0f + 4.0f * rnd());
            const float brk = g_ragdollBreak > 0.0f ? g_ragdollBreak : kStiffInf;
            if (g_spawnKind == 5)
                scenes::addPoseRagdoll(frag, at, scale, brk, randomShirt());
            else
                scenes::addRagdoll(frag, at, scale, brk, randomShirt());
        }
        g_world.spawnScene(frag, Vec2());
        extendColors();
        return;
    }
    std::vector<SpawnDesc2D> d((size_t)count);
    for (int k = 0; k < count; k++)
    {
        SpawnDesc2D &o = d[(size_t)k];
        setSpawnShape(o, g_spawnKind);
        o.pos = Vec2(v.x0 + (v.x1 - v.x0) * rnd(), v.y1 + 1.0f + 2.0f * rnd());
        o.angle = kPi * rnd();
        o.density = g_spawnDensity;
        o.friction = g_spawnFriction;
    }
    g_world.spawn(d.data(), count);
    extendColors();
}

// ---------------------------------------------------------------------------------------
// Instances and overlays
// ---------------------------------------------------------------------------------------
static void addLine(std::vector<Instance2D> &out, Vec2 a, Vec2 b, float thick, uint32_t rgba)
{
    const Vec2 d = b - a;
    const float len = length(d);
    if (len < 1e-6f)
        return;
    Instance2D o;
    o.x = 0.5f * (a.x + b.x);
    o.y = 0.5f * (a.y + b.y);
    o.angle = std::atan2(d.y, d.x);
    o.hx = 0.5f * len;
    o.hy = 0.5f * thick;
    o.rgba = rgba;
    o.shape = 0;
    o.flags = 0;
    out.push_back(o);
}

static void addDot(std::vector<Instance2D> &out, Vec2 p, float r, uint32_t rgba)
{
    Instance2D o;
    o.x = p.x;
    o.y = p.y;
    o.angle = 0.0f;
    o.hx = r;
    o.hy = r;
    o.rgba = rgba;
    o.shape = 1;
    o.flags = 0;
    out.push_back(o);
}

static bool inView(const View &v, Vec2 p, float margin)
{
    return p.x + margin >= v.x0 && p.x - margin <= v.x1 && p.y + margin >= v.y0 && p.y - margin <= v.y1;
}

// One body as renderer primitives: boxes and centred circles are one instance, an offset
// circle is a circle moved to its centre, a capsule one capsule instance, a polygon a fan of
// triangles (a rounded one with its vertices pushed out along the bisectors, so the corners
// are sharp but the size is right), segments and chains a line.
static void addShape(std::vector<Instance2D> &out, const Scene2D &s, int shape, Vec2 p, float a, uint32_t rgba,
                     uint32_t flags)
{
    const size_t k = (size_t)shape;
    const Vec2 h = s.shHalf[k];
    const float r = s.shRad[k];
    const int n = s.shVCnt[k];
    const Vec2 *v = s.verts.data() + s.shVOff[k];
    Instance2D o;
    o.x = p.x;
    o.y = p.y;
    o.angle = a;
    o.hx = h.x;
    o.hy = h.y;
    o.rgba = rgba;
    o.shape = kInstBox;
    o.flags = flags;
    switch (s.shType[k])
    {
    case SHAPE2D_CIRCLE:
        o.shape = kInstCircle;
        if (n > 0)
        {
            const Vec2 c = p + rotate(a, v[0]);
            o.x = c.x;
            o.y = c.y;
            o.hx = o.hy = r;
        }
        break;
    case SHAPE2D_CAPSULE:
    {
        // Authored along local x about the origin; a general core is placed by its midpoint.
        const Vec2 m = (v[0] + v[1]) * 0.5f, d = v[1] - v[0];
        const Vec2 c = p + rotate(a, m);
        o.x = c.x;
        o.y = c.y;
        o.angle = a + std::atan2(d.y, d.x);
        o.hx = 0.5f * length(d);
        o.hy = r;
        o.shape = kInstCapsule;
        break;
    }
    case SHAPE2D_POLYGON:
    {
        Vec2 q[kMaxPolygonVertices];
        const Vec2 *nr = s.norms.data() + s.shVOff[k];
        for (int j = 0; j < n; j++)
        {
            q[j] = v[j];
            if (r > 0.0f)
            {
                const Vec2 mid = nr[(j + n - 1) % n] + nr[j];
                q[j] = v[j] + mid * (r * 2.0f / lengthSq(mid)); // the offset lines' intersection
            }
        }
        o.shape = kInstTriangle;
        for (int j = 1; j + 1 < n; j++)
        {
            o.hx = q[0].x;
            o.hy = q[0].y;
            o.ex[0] = q[j].x;
            o.ex[1] = q[j].y;
            o.ex[2] = q[j + 1].x;
            o.ex[3] = q[j + 1].y;
            o.flags = flags | 0x200u | (j == 1 ? 0x100u : 0u) | (j + 2 == n ? 0x400u : 0u);
            out.push_back(o);
        }
        return;
    }
    case SHAPE2D_SEGMENT:
    case SHAPE2D_CHAIN:
    {
        const int b = s.shType[k] == SHAPE2D_CHAIN ? 1 : 0;
        const Vec2 p1 = p + rotate(a, v[b]), p2 = p + rotate(a, v[b + 1]);
        const Vec2 d = p2 - p1;
        o.x = 0.5f * (p1.x + p2.x);
        o.y = 0.5f * (p1.y + p2.y);
        o.angle = std::atan2(d.y, d.x);
        o.hx = 0.5f * length(d) + 0.04f;
        o.hy = 0.04f;
        break;
    }
    default:
        break;
    }
    out.push_back(o);
}

// Every shape of body i, placed by the body's pose and the shape's own offset and angle.
static void addBody(std::vector<Instance2D> &out, const Scene2D &s, int i, Vec2 p, float a, uint32_t rgba,
                    uint32_t flags)
{
    const int f = s.shapeFirst[(size_t)i];
    for (int k = f; k < f + s.shapeCount[(size_t)i]; k++)
    {
        const Vec2 off = s.shOff[(size_t)k];
        const bool plain = off.x == 0.0f && off.y == 0.0f && s.shAng[(size_t)k] == 0.0f;
        addShape(out, s, k, plain ? p : p + rotate(a, off), a + s.shAng[(size_t)k], rgba, flags);
    }
}

static std::vector<ContactDebug2D> g_contacts;
static bool g_contactsStale = true;

// Event markers: a short-lived dot where a contact began, hit, or a sensor gained or lost a visitor.
// Off (and the device event pass off with it) except in the Sensor Funnel scene.
static constexpr int kEventMarkLife = 30;
static constexpr size_t kEventMarkCap = 4000;

static void collectEventMarks()
{
    for (EventMark &m : g_eventMarks)
        m.life--;
    g_eventMarks.erase(std::remove_if(g_eventMarks.begin(), g_eventMarks.end(), [](const EventMark &m) { return m.life <= 0; }),
                       g_eventMarks.end());
    if (!g_showEvents)
        return;
    const Scene2D &s = g_world.scene();
    const Vec2 *pos = g_world.positions();
    const int bodies = g_world.bodyCount();
    auto shapeP = [&](int shape, Vec2 &p) {
        if (shape < 0 || shape >= s.shapeTotal())
            return false;
        const int b = s.shBody[(size_t)shape];
        if (b < 0 || b >= bodies)
            return false;
        p = pos[b];
        return true;
    };
    auto add = [&](Vec2 p, int kind) {
        if (g_eventMarks.size() < kEventMarkCap)
            g_eventMarks.push_back({p, kEventMarkLife, kind});
    };
    const Events2D &ev = g_world.events();
    Vec2 p;
    for (const ContactEvent2D &e : ev.contactBegin)
        if (shapeP(e.shapeB, p))
            add(p, 0);
    for (const HitEvent2D &e : ev.hit)
        add(e.point, 1);
    for (const SensorEvent2D &e : ev.sensorBegin)
        if (shapeP(e.visitorShape, p))
            add(p, 2);
    for (const SensorEvent2D &e : ev.sensorEnd)
        if (shapeP(e.visitorShape, p))
            add(p, 3);
}

static void buildInstances(std::vector<Instance2D> &out)
{
    const Scene2D &s = g_world.scene();
    const int n = g_world.bodyCount();
    const Vec2 *pos = g_world.positions();
    const float *ang = g_world.angles();
    const bool extras = g_colorMode != 0;
    const Vec2 *vel = g_world.velocities();
    const uint8_t *awake = g_world.awakeFlags();
    const int *gcol = g_world.graphColors();
    const View v = viewRect();
    out.clear();

    for (int i = 0; i < n; i++)
    {
        const Vec2 h = s.half[i];
        if (!inView(v, pos[i], std::max(h.x, h.y) * 1.5f))
            continue;
        uint32_t rgba = g_colors[(size_t)i];
        if (extras && s.mass[i] > 0.0f)
        {
            if (g_colorMode == 1)
                rgba = awake[i] ? packRGB(0.35f, 0.9f, 0.4f) : packRGB(0.25f, 0.45f, 0.95f);
            else if (g_colorMode == 2)
            {
                const float t = std::min(length(vel[i]) / 8.0f, 1.0f);
                rgba = packRGB(0.15f + 0.85f * t, 0.3f * (1.0f - t) + 0.2f, 0.9f * (1.0f - t) + 0.1f);
            }
            else if (g_colorMode == 3)
                rgba = hsv(0.0833f * (float)(gcol[i] & 63), 0.65f, 0.95f);
        }
        else if (extras)
            rgba = kStaticColor;
        addBody(out, s, i, pos[i], ang[i], rgba, g_world.isGrabbed(i) ? 1u : 0u);
    }

    const float unit = g_camHeight * 0.001f; // one overlay "pixel", roughly

    if (g_showGrid)
    {
        const float cs = g_world.cellSize();
        const int x0 = (int)std::floor(v.x0 / cs), x1 = (int)std::ceil(v.x1 / cs);
        const int y0 = (int)std::floor(v.y0 / cs), y1 = (int)std::ceil(v.y1 / cs);
        if (x1 - x0 <= 400 && y1 - y0 <= 400)
        {
            const uint32_t c = packRGB(0.25f, 0.28f, 0.35f);
            for (int x = x0; x <= x1; x++)
                addLine(out, Vec2((float)x * cs, v.y0), Vec2((float)x * cs, v.y1), unit, c);
            for (int y = y0; y <= y1; y++)
                addLine(out, Vec2(v.x0, (float)y * cs), Vec2(v.x1, (float)y * cs), unit, c);
        }
    }

    if (g_showJoints)
    {
        const uint32_t line = packRGB(0.95f, 0.35f, 0.9f), dot = packRGB(0.3f, 1.0f, 1.0f);
        int drawn = 0;
        for (int j = 0; j < s.jointCount() && drawn < 40000; j++)
        {
            const int a = s.jointA[(size_t)j], b = s.jointB[(size_t)j];
            const Vec2 pa = a >= 0 ? pos[a] + rotate(ang[a], s.jointRA[(size_t)j]) : s.jointRA[(size_t)j];
            const Vec2 pb = pos[b] + rotate(ang[b], s.jointRB[(size_t)j]);
            if (!inView(v, pb, 1.0f))
                continue;
            if (a >= 0)
                addLine(out, pos[a], pa, unit * 1.5f, line);
            addLine(out, pos[b], pb, unit * 1.5f, line);
            addDot(out, pb, unit * 3.0f, dot);
            drawn++;
        }
    }

    if (g_showEvents)
    {
        static const uint32_t kinds[4] = {packRGB(0.3f, 1.0f, 0.4f), packRGB(1.0f, 0.6f, 0.1f),
                                          packRGB(0.2f, 0.9f, 1.0f), packRGB(0.5f, 0.5f, 1.0f)};
        for (const EventMark &m : g_eventMarks)
            if (inView(v, m.p, 1.0f))
                addDot(out, m.p, unit * (3.0f + 6.0f * (float)m.life / (float)kEventMarkLife), kinds[m.kind]);
    }

    if (g_showContacts || g_showNormals)
    {
        if (g_contactsStale)
        {
            g_world.downloadContacts(g_contacts);
            g_contactsStale = false;
        }
        int drawn = 0;
        for (const ContactDebug2D &c : g_contacts)
        {
            if (drawn >= 60000)
                break;
            if (!inView(v, c.p, 1.0f))
                continue;
            if (g_showContacts)
                addDot(out, c.p, unit * 3.5f, packRGB(1.0f, 0.2f, 0.2f));
            if (g_showNormals)
            {
                const float f = std::min(c.force / 200.0f, 1.0f);
                const float len = g_camHeight * 0.02f * (0.4f + 1.6f * f);
                addLine(out, c.p, c.p + c.n * len, unit * 1.5f, packRGB(1.0f, 0.9f - 0.7f * f, 0.2f));
            }
            drawn++;
        }
    }
}

static bool posesFinite()
{
    const Vec2 *pos = g_world.positions();
    const float *ang = g_world.angles();
    for (int i = 0; i < g_world.bodyCount(); i++)
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) || !std::isfinite(ang[i]))
            return false;
    return true;
}

// ---------------------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------------------
static void timingRow(const char *name, float ms, float total)
{
    char label[64];
    snprintf(label, sizeof(label), "%.3f ms", ms);
    ImGui::ProgressBar(total > 0.0f ? std::min(ms / total, 1.0f) : 0.0f, ImVec2(120, 0), label);
    ImGui::SameLine();
    ImGui::TextUnformatted(name);
}

static void tip(const char *text)
{
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", text);
}

// One of the mutually exclusive "colour the bodies by ..." overlays.
static void overlayToggle(const char *label, int mode, const char *help)
{
    bool on = g_colorMode == mode;
    if (ImGui::Checkbox(label, &on))
        g_colorMode = on ? mode : 0;
    tip(help);
}

// The panel: scene and transport, then the foldouts, all shut by default.
static void drawUI()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 640), ImGuiCond_FirstUseEver);
    ImGui::Begin("AVBD 2D");
    SolverParams2D &p = g_world.params();
    const WorldStats2D st = g_world.stats();
    const StepTiming2D &tm = g_world.timing();

    if (g_warning)
        ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "PAUSED: %s", g_warning);

    const std::string sceneLabel = g_inCapture ? "capture: " + g_captureName : std::string(g_scenes[g_sceneIndex].name);
    if (ImGui::BeginCombo("Scene", sceneLabel.c_str()))
    {
        for (int i = 0; i < g_sceneCount; i++)
            if (ImGui::Selectable(g_scenes[i].name, !g_inCapture && i == g_sceneIndex))
                loadScene(i);
        ImGui::EndCombo();
    }
    if (ImGui::Button(g_paused ? "Play" : "Pause"))
    {
        g_paused = !g_paused;
        g_warning = nullptr;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        g_paused = true;
        g_stepOnce = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
        restartScene();

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Captures"))
    {
        if (ImGui::Button("Capture now (P)"))
            captureNow();
        ImGui::SameLine();
        if (ImGui::Button("Rescan"))
            refreshCaptureList();
        tip("Saves every body's pose and velocity, the joints that are still intact and the solver\n"
            "parameters to captures2d/. Removed bodies and fractured joints are dropped. Sleep state and\n"
            "warm-start data are not saved: the replay re-settles in a few steps.");
        if (!g_captureStatus.empty())
            ImGui::TextWrapped("%s", g_captureStatus.c_str());
        if (g_inCapture)
            ImGui::Text("from '%s' at step %llu", g_captureMeta.source.c_str(), (unsigned long long)g_captureMeta.step);
        for (const std::string &f : g_captureFiles)
            if (ImGui::Selectable(f.c_str(), g_inCapture && f == g_captureName))
                loadCapture(std::string(kCaptureDir) + "/" + f);
        if (g_captureFiles.empty())
            ImGui::TextDisabled("(no captures yet)");
    }

    if (ImGui::CollapsingHeader("Solver"))
    {
        ImGui::SliderInt("Iterations", &p.iterations, 1, 64);
        tip("AVBD iterations per step: a primal sweep over the graph colours, then the dual update.\n"
            "Cost is linear in this; a low count leaves constraint error instead of exploding.");
        ImGui::SliderFloat("Alpha", &p.alpha, 0.0f, 1.0f);
        tip("Stabilisation (Eq. 18): only (1 - alpha) of the error present at the start of a step is\n"
            "corrected in it. 0 corrects everything at once and makes a deep pile pop.");
        ImGui::SliderFloat("Gamma", &p.gamma, 0.0f, 1.0f);
        tip("Warm-start decay (Eq. 19) of the stiffness and multipliers carried between steps.");
        ImGui::SliderFloat("Beta linear", &p.betaLin, 1.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        tip("Penalty ramp rate for linear rows (contacts, joint anchors).");
        ImGui::SliderFloat("Gravity", &p.gravity.y, -40.0f, 0.0f);

        bool neverSleep = !p.sleepEnabled;
        if (ImGui::Checkbox("Never sleep", &neverSleep))
            p.sleepEnabled = !neverSleep; // turning sleep off wakes everything on the next step
        tip("Solve every dynamic body every step, to A/B the simulation against itself.");
        ImGui::BeginDisabled(neverSleep);
        ImGui::SliderInt("Sleep frames", &p.sleepFrames, 0, 120);
        tip("Steps in the rest window before a body may freeze. 0 disables sleeping.");
        ImGui::SliderFloat("Sleep travel", &p.sleepDisp, 0.0001f, 0.05f, "%.4f m", ImGuiSliderFlags_Logarithmic);
        tip("Net movement allowed across the whole window. This is the rest test;\n"
            "the speed thresholds are only a fast reject.");
        ImGui::SliderFloat("Sleep speed", &p.sleepLinVel, 0.01f, 2.0f, "%.2f m/s", ImGuiSliderFlags_Logarithmic);
        tip("Fast reject, applied at 2x, and the speed at which a mover wakes what it touches.");
        ImGui::SliderFloat("Sleep spin", &p.sleepAngVel, 0.01f, 4.0f, "%.2f rad/s", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        if (ImGui::Button("Wake all (W)"))
            g_world.wakeAll();

        ImGui::Checkbox("Kill fallen bodies", &p.killFallenEnabled);
        tip("Removes and stops simulating anything that falls below the Y threshold below.");
        ImGui::BeginDisabled(!p.killFallenEnabled);
        ImGui::SliderFloat("Kill Y", &p.killY, -1000.0f, 0.0f, "%.0f m");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Kill above speed", &p.maxSpeed, 50.0f, 5000.0f, "%.0f m/s");
        tip("A body faster than this is treated as exploded and removed.");
    }

    if (ImGui::CollapsingHeader("Fidelity"))
    {
        bool capOn = p.contactCap > 0;
        if (ImGui::Checkbox("Contact cap", &capOn))
            p.contactCap = capOn ? 1 : 0;
        tip("Keep at most N contact points per manifold, the deepest first. A 2D manifold holds at\n"
            "most 2 points, so the only cap that changes anything is 1; a box then rests on one corner\n"
            "at a time and can rock. Halves the contact work in a pile.");
        // The slider stores what it shows: 2 is a valid (no-op) cap, not "off", so dragging it up
        // never unticks the checkbox and disables the slider under the cursor.
        ImGui::BeginDisabled(!capOn);
        int pts = std::min(std::max(p.contactCap, 1), 2);
        if (ImGui::SliderInt("Points per manifold", &pts, 1, 2, pts >= 2 ? "2 (all)" : "%d") && capOn)
            p.contactCap = pts;
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Grab"))
    {
        ImGui::SliderFloat("Grab strength", &p.grabStrength, 0.05f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        tip("Multiplier on the mouse-drag spring stiffness.");
        ImGui::SliderFloat("Grab max speed", &p.grabMaxSpeed, 1.0f, 200.0f, "%.0f m/s", ImGuiSliderFlags_Logarithmic);
        tip("Fastest the grab moves a held body. The cursor can run ahead; the body follows at up to\n"
            "this speed, and a blocked body is not wound up to be flung when it slips free.");
        ImGui::Checkbox("Grab connected bodies", &g_grabConnected);
        if (g_grabConnected)
        {
            ImGui::SliderFloat("Grab radius", &g_grabRadius, 0.1f, 50.0f, "%.1f m", ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("Follow joints", &g_grabFollowJoints);
            tip("A jointed body brings what its joints reach, within the radius. A body with no\n"
                "joints (or with this off) brings every body within the radius instead.");
            ImGui::SliderFloat("Falloff", &g_grabFalloff, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Min strength", &g_grabMinStrength, 0.0f, 1.0f, "%.2f");
            ImGui::TextUnformatted("0 falloff holds the whole group equally;");
            ImGui::TextUnformatted("higher concentrates the pull on the click.");
            float curve[48];
            const VkWorld2D::GrabOptions o = grabOptions();
            for (int i = 0; i < 48; i++)
                curve[i] = VkWorld2D::grabWeight(o.radius * (float)i / 47.0f, o);
            ImGui::PlotLines("##grabfalloff", curve, 48, 0, "strength vs distance", 0.0f, 1.0f, ImVec2(0, 45));
        }
        else
            ImGui::TextUnformatted("Single body.");
        if (g_world.grabbing())
            ImGui::Text("holding %d bodies", g_world.grabCount());
    }

    if (ImGui::CollapsingHeader("Spawn"))
    {
        const char *kinds[] = {"Box", "Disc", "Ragdoll", "Capsule", "Polygon", "Ragdoll (A pose)"};
        ImGui::Combo("Object", &g_spawnKind, kinds, 6);
        ImGui::SliderFloat("Spawn size", &g_spawnSize, 0.1f, 4.0f);
        ImGui::SliderFloat("Density", &g_spawnDensity, 0.1f, 20.0f);
        ImGui::SliderFloat("Friction", &g_spawnFriction, 0.0f, 1.5f);
        if (g_spawnKind == 2 || g_spawnKind == 5)
        {
            ImGui::SliderFloat("Ragdoll break force", &g_ragdollBreak, 0.0f, 1500.0f, g_ragdollBreak <= 0.0f ? "never" : "%.0f N");
            tip("Force on a joint's linear multiplier above which the joint breaks. 0 = unbreakable.\n"
                "Density and friction apply to the other shapes; a ragdoll is Box2D's human (density 1, friction 0.2).");
        }
        ImGui::SliderInt("Burst count", &g_spawnCount, 1, 5000);
        if (ImGui::Button("Spawn one (F)"))
            spawnCluster(g_cursor, g_spawnKind, 1, Vec2());
        ImGui::SameLine();
        if (ImGui::Button("Spawn bunch (G)"))
            spawnCluster(g_cursor, g_spawnKind, g_spawnCount, Vec2());
        ImGui::Checkbox("Rain", &g_rain);
        ImGui::SliderFloat("Rain rate (/s)", &g_rainRate, 10.0f, 5000.0f);
    }

    if (ImGui::CollapsingHeader("Camera & controls"))
    {
        ImGui::TextUnformatted("LMB drag  RMB/MMB pan  wheel zoom");
        ImGui::TextUnformatted("F spawn one  G spawn bunch  W wake all");
        ImGui::TextUnformatted("Space pause  . step  R reset  Esc quit");
    }

    if (ImGui::CollapsingHeader("Display"))
    {
        if (ImGui::Checkbox("VSync", &g_vsync))
            g_swapDirty = true;
    }

    if (ImGui::CollapsingHeader("Debug overlays"))
    {
        overlayToggle("Sleeping bodies", 1,
                      "Blue: asleep, and dropped from the constraint graph. Green: awake.");
        overlayToggle("Speed", 2, "Linear speed. Shows the creep a settled-looking pile is hiding.");
        overlayToggle("Graph colors", 3,
                      "Jones-Plassmann colouring of the constraint graph. Bodies sharing a hue share no\n"
                      "constraint and are solved in the same dispatch.");
        ImGui::Checkbox("Joints", &g_showJoints);
        tip("A line from each jointed body's centre to its joint anchor.");
        ImGui::Checkbox("Broadphase grid", &g_showGrid);
        tip("The hash-grid cells the broadphase bins bodies into.");
    }

    if (ImGui::CollapsingHeader("Contact markers"))
    {
        if (ImGui::Checkbox("Contact points", &g_showContacts))
            g_contactsStale = true;
        if (ImGui::Checkbox("Contact normals / load", &g_showNormals))
            g_contactsStale = true;
        tip("A needle along each contact normal, brighter and longer with the normal force.");
        if (ImGui::Checkbox("Event markers", &g_showEvents))
        {
            g_world.setEventsEnabled(g_showEvents);
            g_eventMarks.clear();
        }
        tip("Green: contact begin. Orange: hit. Cyan / blue: sensor overlap begin / end.");
    }

    if (ImGui::CollapsingHeader("Scene stats"))
    {
        ImGui::Text("device    %s", g_dev->deviceName());
        ImGui::Text("bodies    %d (%d awake, %d asleep)", st.bodies, st.awake, st.asleep);
        ImGui::Text("static %d  dynamic %d  removed %d", st.statics, st.dynamic, st.removed);
        ImGui::Text("joints    %d   broken %d", st.joints, st.brokenJoints);
        ImGui::Text("pairs     %d / %d", g_lastResult.pairs, st.pairCap);
        ImGui::Text("contacts  %d / %d", g_lastResult.contacts, st.contactCap);
        ImGui::Text("colors    %d", g_lastResult.colors);
        ImGui::Text("reruns    cap %d  jp %d  colour %d   submits %d", g_lastResult.capacityReruns,
                    g_lastResult.jpReruns, g_lastResult.colorReruns, g_lastResult.submits);
        ImGui::Text("grid cell %.2f m", g_world.cellSize());
        ImGui::Text("device mem %.1f MB", (double)st.deviceBytes / (1024.0 * 1024.0));
        if (!g_lastResult.ok)
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "capacity ceiling hit: %s", g_lastResult.failResource);
    }

    if (ImGui::CollapsingHeader("Timings"))
    {
        if (tm.valid)
        {
            ImGui::Text("GPU total %.3f ms", tm.gpuMs);
            timingRow("joint init + integrate", tm.integrateMs, tm.gpuMs);
            timingRow("broadphase + sort", tm.broadMs, tm.gpuMs);
            timingRow("narrowphase + wake", tm.narrowMs, tm.gpuMs);
            timingRow("graph + colouring", tm.buildMs, tm.gpuMs);
            timingRow("primal (all iterations)", tm.primalMs, tm.gpuMs);
            timingRow("dual (all iterations)", tm.dualMs, tm.gpuMs);
            timingRow("velocity, guard, sleep, copy", tm.tailMs, tm.gpuMs);
        }
        else
            ImGui::TextUnformatted("GPU timestamps unavailable");
        float maxHost = 1.0f;
        for (int i = 0; i < kHistory; i++)
            maxHost = std::max(maxHost, g_histHost[i]);
        ImGui::PlotLines("##host", g_histHost, kHistory, g_histPos, "host step ms", 0.0f, maxHost, ImVec2(0, 50));
        ImGui::PlotLines("##gpu", g_histGpu, kHistory, g_histPos, "GPU ms", 0.0f, maxHost, ImVec2(0, 50));
        ImGui::Text("%.1f fps   step %.2f ms (host)", ImGui::GetIO().Framerate, g_lastResult.stepMs);
    }
    ImGui::End();
}

static void printUsage(FILE *f)
{
    fprintf(f,
            "avbd2d_demo [--scene <index|name>] [--frames N] [--smoke] [--show] [--no-vsync] [--verbose] [--help]\n"
            "  --scene S     start on scene S (index or name); scenes:\n");
    int n = 0;
    const scenes::SceneEntry2D *e = scenes::catalogue(n);
    for (int i = 0; i < n; i++)
        fprintf(f, "                  %d  %s\n", i, e[i].name);
    fprintf(f,
            "                or a capture file (P in the demo writes captures2d/*.a2dcap)\n"
            "  --frames N    run N frames, then exit\n"
            "  --smoke       unattended run of every scene with validation on; the exit code is the gate\n"
            "  --show        show the window during --smoke / --frames (hidden by default)\n"
            "  --no-vsync    present with MAILBOX when available\n"
            "  --verbose     print the selected GPU\n");
}

// ---------------------------------------------------------------------------------------
// Smoke script: per scene, load, run, spawn, grab-and-drag the first dynamic body, turn the
// overlays on, wake everything. Scenes of more than 50k bodies get a short run.
// ---------------------------------------------------------------------------------------
static std::vector<int> g_smokeStart; // first smoke frame of each scene, then the total

static int smokeFramesFor(int scene)
{
    return g_scenes[scene].approxBodies > 50000 ? 15 : 90;
}

static void smokeTick(int smokeFrame)
{
    int scene = 0;
    while (scene + 1 < g_sceneCount && smokeFrame >= g_smokeStart[(size_t)scene + 1])
        scene++;
    const int f = smokeFrame - g_smokeStart[(size_t)scene];
    const int frames = smokeFramesFor(scene);
    if (f == 0)
    {
        loadScene(scene);
        g_colorMode = scene % 4;
    }
    g_world.setExtraReadback(g_colorMode != 0);
    if (frames < 90)
        return;
    int dyn = -1;
    for (int i = 0; i < g_world.bodyCount() && dyn < 0; i++)
        if (g_world.scene().mass[i] > 0.0f)
            dyn = i;
    if (dyn < 0)
        return;
    const Vec2 p = g_world.positions()[dyn];
    if (f == 10)
        spawnCluster(Vec2(p.x, p.y + 6.0f), scene % 3, scene % 3 == 2 ? 6 : 40, Vec2());
    else if (f == 20)
        g_showContacts = g_showNormals = g_showJoints = g_showGrid = true;
    else if (f == 30)
        g_world.grabBegin(p);
    else if (f > 30 && f < 60)
        g_world.grabMove(Vec2(p.x, p.y + 0.15f));
    else if (f == 60)
        g_world.grabEnd();
    else if (f == 70)
    {
        g_showContacts = g_showNormals = g_showJoints = g_showGrid = false;
        g_world.wakeAll();
    }
    else if (f == 80 && scene == 0)
    {
        // Capture the live world and replay the file (the replay keeps running to the scene's end).
        captureNow();
        const std::string path = g_captureStatus.substr(g_captureStatus.rfind("-> ") + 3);
        if (g_captureStatus.rfind("captured", 0) != 0 || !loadCapture(path))
            g_stepFailures++;
        std::error_code ec;
        std::filesystem::remove(path, ec); // the smoke run leaves nothing behind
        refreshCaptureList();
    }
}

// ---------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int startScene = 0;
    std::string startCapture; // --scene <file.a2dcap>
    int benchFrames = 0;
    bool smoke = false, verbose = false, show = false;

    g_scenes = scenes::catalogue(g_sceneCount);
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (!strcmp(a, "--help") || !strcmp(a, "-h"))
        {
            printUsage(stdout);
            return 0;
        }
        else if (!strcmp(a, "--smoke")) smoke = true;
        else if (!strcmp(a, "--show")) show = true;
        else if (!strcmp(a, "--verbose")) verbose = true;
        else if (!strcmp(a, "--no-vsync")) g_vsync = false;
        else if (!strcmp(a, "--frames") && hasValue) benchFrames = atoi(argv[++i]);
        else if (!strcmp(a, "--scene") && hasValue)
        {
            const char *v = argv[++i];
            const size_t vl = strlen(v), el = strlen(kCaptureExt2D);
            if (vl > el && !strcmp(v + vl - el, kCaptureExt2D))
            {
                startCapture = v;
                continue;
            }
            bool found = false;
            for (int k = 0; k < g_sceneCount; k++)
                if (!strcmp(v, g_scenes[k].name))
                {
                    startScene = k;
                    found = true;
                }
            if (!found)
                startScene = atoi(v);
        }
        else
        {
            fprintf(stderr, "avbd2d_demo: unknown option or missing value: '%s'\n", a);
            printUsage(stderr);
            return 2;
        }
    }
    if (smoke)
        avbdvk::setValidationOverride(true); // the point of the run; before the Device exists

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    // Unattended runs render into a hidden window: same swapchain path, nothing on screen.
    const Uint32 hiddenFlag = (smoke || benchFrames > 0) && !show ? SDL_WINDOW_HIDDEN : 0;
    g_window = SDL_CreateWindow("AVBD 2D", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kWinWidth, kWinHeight,
                                SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | hiddenFlag);
    if (!g_window)
    {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    // The solver's device is the demo's device: configure it for presentation, then init.
    g_dev = &g_world.device();
    g_dev->setVerbose(verbose);
    unsigned int extCount = 0;
    SDL_Vulkan_GetInstanceExtensions(g_window, &extCount, nullptr);
    std::vector<const char *> instExts(extCount);
    SDL_Vulkan_GetInstanceExtensions(g_window, &extCount, instExts.data());
    g_dev->requireInstanceExtensions(instExts);
    g_dev->enablePresentation([](VkInstance instance) -> VkSurfaceKHR {
        VkSurfaceKHR surf = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(g_window, instance, &surf))
            fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        return surf;
    });
    g_world.init();
    g_surface = g_dev->surface();

    g_swapFormat = chooseSurfaceFormat().format;
    createRenderPass();
    if (!createSwapchain())
    {
        fprintf(stderr, "avbd2d_demo: initial swapchain creation failed (window minimised?)\n");
        return 1;
    }
    g_renderer.init(*g_dev, g_renderPass);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = g_dev->queueFamily();
    VK_CHECK(vkCreateCommandPool(g_dev->device(), &poolInfo, nullptr, &g_cmdPool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g_cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(g_dev->device(), &cbai, g_cmdBufs));
    for (int i = 0; i < kFramesInFlight; i++)
    {
        VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(g_dev->device(), &semi, nullptr, &g_imageAvail[i]));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(g_dev->device(), &fci, nullptr, &g_inFlight[i]));
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForVulkan(g_window);
    ImGui_ImplVulkan_InitInfo vii{};
    vii.Instance = g_dev->instance();
    vii.PhysicalDevice = g_dev->physicalDevice();
    vii.Device = g_dev->device();
    vii.QueueFamily = g_dev->queueFamily();
    vii.Queue = g_dev->queue();
    vii.RenderPass = g_renderPass;
    vii.MinImageCount = 2;
    vii.ImageCount = (uint32_t)g_swapImages.size();
    vii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vii.PipelineCache = g_dev->pipelineCache();
    vii.DescriptorPoolSize = 8;
    ImGui_ImplVulkan_Init(&vii);

    loadScene(startScene);
    refreshCaptureList();
    if (!startCapture.empty())
    {
        if (!loadCapture(startCapture))
            return 2;
        // Starting on a capture opens on the view it was saved with.
        g_camCenter = g_captureScene.cameraCenter;
        g_camHeight = g_captureScene.cameraHeight;
    }

    std::vector<Instance2D> instances;
    bool running = true;
    int frameNo = 0;
    int smokeFrame = 0;
    int smokeFrames = 0;
    if (smoke)
    {
        g_smokeStart.assign((size_t)g_sceneCount + 1, 0);
        for (int i = 0; i < g_sceneCount; i++)
            g_smokeStart[(size_t)i + 1] = g_smokeStart[(size_t)i] + smokeFramesFor(i);
        smokeFrames = g_smokeStart[(size_t)g_sceneCount];
    }

    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            ImGui_ImplSDL2_ProcessEvent(&e);
            const bool uiMouse = ImGui::GetIO().WantCaptureMouse;
            const bool uiKeys = ImGui::GetIO().WantCaptureKeyboard;
            switch (e.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                if (e.window.event == SDL_WINDOWEVENT_RESIZED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                    g_swapDirty = true;
                break;
            case SDL_KEYDOWN:
                if (uiKeys || e.key.repeat)
                    break;
                if (e.key.keysym.sym == SDLK_ESCAPE) running = false;
                else if (e.key.keysym.sym == SDLK_r) restartScene();
                else if (e.key.keysym.sym == SDLK_p) captureNow();
                else if (e.key.keysym.sym == SDLK_SPACE) g_paused = !g_paused;
                else if (e.key.keysym.sym == SDLK_PERIOD) g_stepOnce = true;
                else if (e.key.keysym.sym == SDLK_f) spawnCluster(g_cursor, g_spawnKind, 1, Vec2());
                else if (e.key.keysym.sym == SDLK_g) spawnCluster(g_cursor, g_spawnKind, g_spawnCount, Vec2());
                else if (e.key.keysym.sym == SDLK_w) g_world.wakeAll();
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (uiMouse)
                    break;
                if (e.button.button == SDL_BUTTON_LEFT)
                    g_world.grabBegin(screenToWorld(e.button.x, e.button.y), grabOptions());
                else if (e.button.button == SDL_BUTTON_RIGHT || e.button.button == SDL_BUTTON_MIDDLE)
                    g_panning = true;
                break;
            case SDL_MOUSEBUTTONUP:
                if (e.button.button == SDL_BUTTON_LEFT)
                    g_world.grabEnd();
                else if (e.button.button == SDL_BUTTON_RIGHT || e.button.button == SDL_BUTTON_MIDDLE)
                    g_panning = false;
                break;
            case SDL_MOUSEMOTION:
                g_cursor = screenToWorld(e.motion.x, e.motion.y);
                if (g_world.grabbing())
                    g_world.grabMove(g_cursor);
                if (g_panning)
                {
                    int wh = 1;
                    SDL_GetWindowSize(g_window, nullptr, &wh);
                    const float perPixel = g_camHeight / (float)std::max(wh, 1);
                    g_camCenter.x -= (float)e.motion.xrel * perPixel;
                    g_camCenter.y += (float)e.motion.yrel * perPixel;
                }
                break;
            case SDL_MOUSEWHEEL:
                if (!uiMouse)
                    g_camHeight = std::min(std::max(g_camHeight * std::pow(0.9f, (float)e.wheel.y), 2.0f), 400.0f);
                break;
            default:
                break;
            }
        }

        if (smoke)
            smokeTick(smokeFrame);

        if (g_swapDirty && g_swapchain)
        {
            vkDeviceWaitIdle(g_dev->device());
            createSwapchain();
            g_swapDirty = false;
        }

        // --- Simulate ---------------------------------------------------------------------
        if (!g_paused || g_stepOnce)
        {
            if (g_rain && !g_paused)
            {
                g_rainCarry += g_rainRate * g_world.params().dt;
                const int n = (int)g_rainCarry;
                g_rainCarry -= (float)n;
                spawnRain(n);
            }
            g_world.setExtraReadback(g_colorMode != 0);
            g_lastResult = g_world.step();
            g_stepsTaken++;
            collectEventMarks();
            g_contactsStale = true;
            g_histHost[g_histPos] = g_lastResult.stepMs;
            g_histGpu[g_histPos] = g_world.timing().valid ? g_world.timing().gpuMs : 0.0f;
            g_histPos = (g_histPos + 1) % kHistory;
            if (!g_lastResult.ok)
            {
                g_stepFailures++;
                if (!smoke)
                {
                    // A diverged simulation would keep asking for more memory: stop it.
                    g_paused = true;
                    g_warning = "a capacity ceiling was hit (the scene diverged?) - press R to reset";
                }
            }
            if (!posesFinite())
            {
                g_nonFinite++;
                if (!smoke)
                {
                    g_paused = true;
                    g_warning = "non-finite poses - press R to reset";
                }
            }
            g_stepOnce = false;
        }

        // --- Render -----------------------------------------------------------------------
        vkWaitForFences(g_dev->device(), 1, &g_inFlight[g_frameIdx], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(g_dev->device(), g_swapchain, UINT64_MAX, g_imageAvail[g_frameIdx],
                                             VK_NULL_HANDLE, &imageIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        {
            vkDeviceWaitIdle(g_dev->device());
            createSwapchain();
            continue;
        }
        vkResetFences(g_dev->device(), 1, &g_inFlight[g_frameIdx]);

        VkCommandBuffer cmd = g_cmdBufs[g_frameIdx];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        VkClearValue clear{};
        clear.color = {{0.10f, 0.11f, 0.13f, 1.0f}};
        VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpbi.renderPass = g_renderPass;
        rpbi.framebuffer = g_framebuffers[imageIndex];
        rpbi.renderArea = {{0, 0}, g_swapExtent};
        rpbi.clearValueCount = 1;
        rpbi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport{0, 0, (float)g_swapExtent.width, (float)g_swapExtent.height, 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, g_swapExtent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        float cam[4];
        camera(cam);
        buildInstances(instances);
        g_renderer.draw(cmd, g_frameIdx, instances.data(), (uint32_t)instances.size(), cam);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        drawUI();
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &g_imageAvail[g_frameIdx];
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &g_renderDone[imageIndex];
        VK_CHECK(vkQueueSubmit(g_dev->queue(), 1, &si, g_inFlight[g_frameIdx]));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &g_renderDone[imageIndex];
        pi.swapchainCount = 1;
        pi.pSwapchains = &g_swapchain;
        pi.pImageIndices = &imageIndex;
        VkResult pres = vkQueuePresentKHR(g_dev->queue(), &pi);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR)
        {
            vkDeviceWaitIdle(g_dev->device());
            createSwapchain();
        }
        g_frameIdx = (g_frameIdx + 1) % kFramesInFlight;

        frameNo++;
        smokeFrame++;
        if (benchFrames > 0 && frameNo >= benchFrames)
            break;
        if (smoke && smokeFrame >= smokeFrames)
            break;
    }

    int exitCode = 0;
    if (g_stepFailures > 0 || g_nonFinite > 0)
        exitCode = 1;
    if (smoke)
    {
        const uint32_t vmsgs = avbdvk::Device::validationMessageCount();
        printf("smoke: scenes %d frames %d step-failures %d non-finite-steps %d validation-messages %u\n",
               g_sceneCount, frameNo, g_stepFailures, g_nonFinite, vmsgs);
        if (vmsgs > 0)
            exitCode = 1;
    }

    vkDeviceWaitIdle(g_dev->device());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    g_renderer.destroy();
    for (int i = 0; i < kFramesInFlight; i++)
    {
        vkDestroySemaphore(g_dev->device(), g_imageAvail[i], nullptr);
        vkDestroyFence(g_dev->device(), g_inFlight[i], nullptr);
    }
    vkDestroyCommandPool(g_dev->device(), g_cmdPool, nullptr);
    destroySwapchainObjects();
    vkDestroySwapchainKHR(g_dev->device(), g_swapchain, nullptr);
    vkDestroyRenderPass(g_dev->device(), g_renderPass, nullptr);

    // destroy() also shuts the shared device down, so it comes last.
    g_world.destroy();
    SDL_DestroyWindow(g_window);
    SDL_Quit();
    return exitCode;
}
