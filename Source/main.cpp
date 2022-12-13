#include <CL/sycl.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <stdlib.h> 
#include <cmath>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define __cdecl
#define __stdcall


using namespace sycl;

constexpr size_t boids_size = 1000;

typedef std::vector<int> IntVector;
typedef std::array<float, boids_size> FloatArray;


struct triangle
{
    float p1_x, p1_y, p2_x, p2_y, p3_x, p3_y;
};

std::string vertexShader =
"#version 330 core\n"
"\n"
"layout(location = 0) in vec4 position;\n"
"uniform mat4 u_MVP;\n"
"\n"
"void main()\n"
"{\n"
"   gl_Position = u_MVP * position; \n"
"}\n";

std::string fragmentShader =
"#version 330 core\n"
"\n"
"layout(location = 0) out vec4 color;"
"\n"
"void main()\n"
"{\n"
"   color = vec4(0.0, 1.0, 0.0, 1.0);\n"
"}\n";

constexpr float visual_range = 100.0f;
constexpr float protected_range = 20.0f;

constexpr float turnfactor = 0.017f;
constexpr float centering_factor = 0.000013f;
constexpr float avoid_factor = 0.0015f;
constexpr float align_factor = 0.01f;
constexpr float maxspeed = 1.2f;
constexpr float minspeed = 0.9f;


constexpr float windowWidth = 1600.0f;  // 1920
constexpr float windowHeight = 900.0f;  // 1080

constexpr float margin = 200.0f;
constexpr float leftmargin = margin;
constexpr float rightmargin = windowWidth - margin;
constexpr float topmargin = windowHeight - margin;
constexpr float bottommargin = margin;


static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};


void RenderFrame(queue& q, range<1> &num_items,
    float* gpu_x, float* gpu_y, float* gpu_vx, float* gpu_vy, float* gpu_p1_x, float* gpu_p1_y, float* gpu_p2_x, float* gpu_p2_y, float* gpu_p3_x, float* gpu_p3_y)
{
    q.submit([&](handler& h) {

    h.parallel_for(num_items, [=](auto i) {
    float x = gpu_x[i];
    float y = gpu_y[i];
    float vx = gpu_vx[i];
    float vy = gpu_vy[i];

    auto distance = [&](float x1, float y1, float x2, float y2)
    {
        return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    };
    // NEW VELOCITY
    // naive iteration
    float close_dx = 0.0f;
    float close_dy = 0.0f;

    float vx_avg = 0.0f;
    float vy_avg = 0.0f;

    float x_avg = 0.0f;
    float y_avg = 0.0f;

    unsigned int neighbors = 0;

    for (int j = 0; j < boids_size; j++)
    {
        if (j == i) continue;
        float friend_x = gpu_x[j];
        float friend_y = gpu_y[j];
        float dist = distance(x, y, friend_x, friend_y);
        if (dist > visual_range)
            continue;

        float friend_vx = gpu_vx[j];
        float friend_vy = gpu_vy[j];

        if (dist < protected_range)
        {
            close_dx += x - friend_x;
            close_dy += y - friend_y;
            continue;
        }
        neighbors++;
        vx_avg += friend_vx;
        vy_avg += friend_vy;
        x_avg += friend_x;
        y_avg += friend_y;
    }

    if (neighbors > 0)
    {
        // alignment
        vx_avg /= neighbors;
        vy_avg /= neighbors;
        vx += (vx_avg - vx) * align_factor;
        vy += (vy_avg - vy) * align_factor;

        // cohesion
        x_avg /= neighbors;
        y_avg /= neighbors;
        vx += (x_avg - x) * centering_factor;
        vy += (y_avg - y) * centering_factor;
    }

    // separation
    vx += close_dx * avoid_factor;
    vy += close_dy * avoid_factor;


    // margin
    if (x < leftmargin)
        vx += turnfactor;
    else if (x > rightmargin)
        vx -= turnfactor;
    if (y < bottommargin)
        vy += turnfactor;
    else if (y > topmargin)
        vy -= turnfactor;

    // speed limit
    float speed = sqrt(vx * vx + vy * vy);
    if (speed > maxspeed)
    {
        vx = vx / speed * maxspeed;
        vy = vy / speed * maxspeed;
    }

    if (speed < minspeed)
    {
        vx = vx / speed * minspeed;
        vy = vy / speed * minspeed;
    }

    //fill the triangle
    float vp_x = -gpu_vy[i];
    float vp_y = gpu_vx[i];
    float scale = 2 / sqrt(vp_x * vp_x + vp_y * vp_y);
    vp_x *= scale;
    vp_y *= scale;

    // update velocity
    gpu_vx[i] = vx;
    gpu_vy[i] = vy;

    // update position
    float new_x = x + vx;
    float new_y = y + vy;
    gpu_x[i] = new_x;
    gpu_y[i] = new_y;


    gpu_p1_x[i] = new_x + vp_x;
    gpu_p1_y[i] = new_y + vp_y;
    gpu_p2_x[i] = new_x - vp_x;
    gpu_p2_y[i] = new_y - vp_y;
    gpu_p3_x[i] = new_x + vx * scale * 2.5;
    gpu_p3_y[i] = new_y + vy * scale * 2.5;
        });
        });
}

void InitializeInput(FloatArray &cpu_x, FloatArray &cpu_y,FloatArray &cpu_vx,FloatArray &cpu_vy,FloatArray &cpu_p1_x,
FloatArray &cpu_p1_y, FloatArray &cpu_p2_x,FloatArray &cpu_p2_y,FloatArray &cpu_p3_x,FloatArray &cpu_p3_y)
{
    srand(time(NULL));
    for (size_t i = 0; i < boids_size; i++)
    {
        cpu_x[i] = rand() % static_cast<int>(windowWidth - margin * 2) + margin;
        cpu_y[i] = rand() % static_cast<int>(windowHeight - margin * 2) + margin;
        float randAngle = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 3.141592653589 * 2;
        cpu_vx[i] = minspeed * cos(randAngle);
        cpu_vy[i] = minspeed * sin(randAngle);
        float vp_x = -cpu_vy[i];
        float vp_y = cpu_vx[i];
        float scale = 2 / sqrt(vp_x * vp_x + vp_y * vp_y);
        vp_x *= scale;
        vp_y *= scale;
        
        cpu_p1_x[i] = cpu_x[i] + vp_x;
        cpu_p1_y[i] = cpu_y[i] + vp_y;
        cpu_p2_x[i] = cpu_x[i] - vp_x;
        cpu_p2_y[i] = cpu_y[i] - vp_y;
        cpu_p3_x[i] = cpu_x[i] + cpu_vx[i] * scale * 2.5;
        cpu_p3_y[i] = cpu_y[i] + cpu_vy[i] * scale * 2.5;

    }
}


static unsigned int CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }
    return id;
}

static unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

int main(int argc, char* argv[]) {
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(windowWidth, windowHeight, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    //glewinit after glfwmakeconfextcurrent
    if (glewInit() != GLEW_OK)
        std::cout << "Error";

    unsigned int buf;
    //tworzymy bufor, bindujemy go jako array bufor i przypisujemy dane
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    default_selector d_selector;

    auto handle_async_error = [](exception_list elist) {
        for (auto& e : elist) {
            try { std::rethrow_exception(e); }
            catch (sycl::exception& e) {
                std::cout << "ASYNC EXCEPTION!!\n";
                std::cout << e.what() << "\n";
            }
        }
    };

    queue q(d_selector, exception_handler);
    range<1> num_items{ boids_size };


    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    glm::mat4 proj = glm::ortho(0.0f, (float)windowWidth, 0.0f, (float)windowHeight, -1.0f, 1.0f);
    int location = glGetUniformLocation(shader, "u_MVP");
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(proj));


    FloatArray cpu_x;
    FloatArray cpu_y;
    FloatArray cpu_vx;
    FloatArray cpu_vy;
    FloatArray cpu_p1_x;
    FloatArray cpu_p1_y;
    FloatArray cpu_p2_x;
    FloatArray cpu_p2_y;
    FloatArray cpu_p3_x;
    FloatArray cpu_p3_y;
    //CPUBoids cpu_boids;
    InitializeInput(cpu_x,cpu_y,cpu_vx,cpu_vy, cpu_p1_x, cpu_p1_y,cpu_p2_x,cpu_p2_y,cpu_p3_x,cpu_p3_y);

    // Copy all buffers to GPU memory
    float *gpu_x = malloc_device<float>(boids_size, q);
    float *gpu_y = malloc_device<float>(boids_size, q);
    float* gpu_vx = malloc_device<float>(boids_size, q);
    float* gpu_vy = malloc_device<float>(boids_size, q);
    float* gpu_p1_x = malloc_device<float>(boids_size, q);
    float* gpu_p1_y = malloc_device<float>(boids_size, q);
    float* gpu_p2_x = malloc_device<float>(boids_size, q);
    float* gpu_p2_y = malloc_device<float>(boids_size, q);
    float* gpu_p3_x = malloc_device<float>(boids_size, q);
    float* gpu_p3_y = malloc_device<float>(boids_size, q);

    q.submit([&](handler& h) {
        h.memcpy(gpu_x, &cpu_x[0], boids_size * sizeof(float));}).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_y, &cpu_y[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_vx, &cpu_vx[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_vy, &cpu_vy[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p1_x, &cpu_p1_x[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p1_y, &cpu_p1_y[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p2_x, &cpu_p2_x[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p2_y, &cpu_p2_y[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p3_x, &cpu_p3_x[0], boids_size * sizeof(float)); }).wait();
    q.submit([&](handler& h) {
        h.memcpy(gpu_p3_y, &cpu_p3_y[0], boids_size * sizeof(float)); }).wait();

    //temporary triangle vector TODO: delete it
    std::array<triangle, boids_size> triangles;
    int frame = 0, time, timebase = 0;

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < boids_size; i++)
        {
            triangles[i].p1_x = cpu_p1_x[i];
            triangles[i].p1_y = cpu_p1_y[i];
            triangles[i].p2_x = cpu_p2_x[i];
            triangles[i].p2_y = cpu_p2_y[i];
            triangles[i].p3_x = cpu_p3_x[i];
            triangles[i].p3_y = cpu_p3_y[i];

        }
        //schedule frame to render
        RenderFrame(q, num_items, gpu_x, gpu_y, gpu_vx, gpu_vy, gpu_p1_x, gpu_p1_y, gpu_p2_x, gpu_p2_y, gpu_p3_x, gpu_p3_y);
        //draw
        glBufferData(GL_ARRAY_BUFFER, boids_size * sizeof(triangle), &(triangles[0]), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, boids_size * 3);
        //fps counter
        frame++;
        time = glfwGetTime();
        if (time - timebase > 1000) {
            std::cout << "FPS: " << frame * 1000.0 / (time - timebase) <<std::endl;;
            timebase = time;
            frame = 0;
        }
        glfwSwapBuffers(window);
        glfwPollEvents();

        //wait until frame is rendered and copy rendered frame to host
        q.wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p1_x[0], gpu_p1_x,boids_size * sizeof(float));}).wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p1_y[0], gpu_p1_y,boids_size * sizeof(float)); }).wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p2_x[0], gpu_p2_x,boids_size * sizeof(float)); }).wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p2_y[0], gpu_p2_y, boids_size * sizeof(float)); }).wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p3_x[0], gpu_p3_x, boids_size * sizeof(float)); }).wait();
        q.submit([&](handler& h) {
            h.memcpy(&cpu_p3_y[0], gpu_p3_y, boids_size * sizeof(float)); }).wait();
    }
    glfwTerminate();

    free(gpu_x, q);
    free(gpu_y, q);
    free(gpu_vx, q);
    free(gpu_vy, q);
    free(gpu_p1_x, q);
    free(gpu_p1_y, q);
    free(gpu_p2_x, q);
    free(gpu_p2_y, q);
    free(gpu_p3_x, q);
    free(gpu_p3_y, q);

    return 0;
}