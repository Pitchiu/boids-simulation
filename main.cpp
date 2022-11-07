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
typedef std::vector<int> IntVector;

struct Boid
{
    float x;
    float y;
    float vx;
    float vy;
};
typedef std::vector<Boid> BoidVector;

struct TrianglePosition
{
    float p1_x, p1_y;
    float p2_x, p2_y;
    float p3_x, p3_y;
};
typedef std::vector<TrianglePosition> TrianglePositionVector;

struct PositionedBoid
{
    struct Boid boid;
    struct TrianglePosition triangle;
};
typedef std::vector<PositionedBoid> PositionedBoidVector;

size_t boids_size = 1000;

constexpr float visual_range = 100.0f;
constexpr float protected_range = 20.0f;

constexpr float turnfactor = 0.017f;
constexpr float centering_factor = 0.000013f;
constexpr float avoid_factor = 0.0015f;
constexpr float align_factor = 0.01f;  //0.01
constexpr float maxspeed = 1.2f;
constexpr float minspeed = 0.9f;
//constexpr float max_bias = 0.01;
//constexpr float bias_increment = 0.00004;
//constexpr float bias_val = 0.001;


constexpr float windowWidth = 1600.0f;
constexpr float windowHeight = 900.0f;

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

void RenderFrame(queue& q, const PositionedBoidVector& in_vector, PositionedBoidVector& out_vector)
{
    range<1> num_items{ boids_size };
    buffer boids_buf(in_vector.data(), num_items);
    buffer positioned_boids_buf(out_vector.data(), num_items);

    q.submit([&](handler& h) {
        accessor in(boids_buf, h, read_only);
        accessor out(positioned_boids_buf, h, write_only);

        h.parallel_for(num_items, [=](auto i) {
            const PositionedBoid *elem = &in[i];
            PositionedBoid result;

            float x = elem->boid.x;
            float y = elem->boid.y;
            float vx = elem->boid.vx;
            float vy = elem->boid.vy;

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

            for (int j = 0; j < in.size(); j++)
            {
                if (j == i) continue;
                float friend_x = in[j].boid.x;
                float friend_y = in[j].boid.y;
                float dist = distance(x, y, friend_x, friend_y);
                if(dist>visual_range)
                    continue;

                float friend_vx = in[j].boid.vx;
                float friend_vy = in[j].boid.vy;

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

            // update velocity
            result.boid.vx = vx;
            result.boid.vy = vy;

            // update position
            result.boid.x = x + vx;
            result.boid.y = y + vy;

            //fill the triangle
            TrianglePosition* triangle = &result.triangle;
            float vp_x = -elem->boid.vy;
            float vp_y = elem->boid.vx;
            float scale = 2 / sqrt(vp_x * vp_x + vp_y * vp_y);
            vp_x *= scale;
            vp_y *= scale;

            triangle->p1_x = result.boid.x + vp_x;
            triangle->p1_y = result.boid.y + vp_y;
            triangle->p2_x = result.boid.x - vp_x;
            triangle->p2_y = result.boid.y - vp_y;
            triangle->p3_x = result.boid.x + result.boid.vx * scale * 2.5;
            triangle->p3_y = result.boid.y + result.boid.vy * scale * 2.5;

            out[i] = result;
            });
        });
    q.wait();
}


void InitializeInputVector(PositionedBoidVector& a)
{
    srand(time(NULL));
    for (size_t i = 0; i < a.size(); i++)
    {
        a[i].boid.x = rand() % static_cast<int>(windowWidth-margin*2) + margin;
        a[i].boid.y = rand() % static_cast<int>(windowHeight-margin*2) + margin;
        float randAngle = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 3.141592653589 * 2;
        a[i].boid.vx = minspeed * cos(randAngle);
        a[i].boid.vy = minspeed * sin(randAngle);
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

    TrianglePositionVector triangles_vector;
    PositionedBoidVector output_vector;
    PositionedBoidVector input_vector;

    triangles_vector.resize(boids_size);
    input_vector.resize(boids_size);
    output_vector.resize(boids_size);

    InitializeInputVector(input_vector);

    unsigned int buf;
    //tworzymy bufor, bindujemy go jako array bufor i przypisujemy dane
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    default_selector d_selector;
    queue q(d_selector, exception_handler);

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

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    glm::mat4 proj = glm::ortho(0.0f, (float)windowWidth, 0.0f, (float)windowHeight, -1.0f, 1.0f);
    int location = glGetUniformLocation(shader, "u_MVP");
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(proj));
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        RenderFrame(q, input_vector, output_vector);

        for (int i = 0; i < boids_size; i++)
            triangles_vector[i] = output_vector[i].triangle;
        glBufferData(GL_ARRAY_BUFFER, boids_size * sizeof(TrianglePosition), triangles_vector.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, boids_size * 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
        input_vector = output_vector;
    }
    glfwTerminate();

    triangles_vector.clear();
    output_vector.clear();
    input_vector.clear();

    return 0;
}