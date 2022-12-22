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

constexpr size_t boids_size = 8192;


typedef std::vector<int> IntVector;
typedef std::array<float, boids_size> FloatArray;

struct Positions
{
    float x[boids_size];
    float y[boids_size];
};

struct Velocities
{
    float vx[boids_size];
    float vy[boids_size];
};

struct TrianglePositions
{
    float p1_x;
    float p1_y;
    float p2_x;
    float p2_y;
    float p3_x;
    float p3_y;
};

struct Boids
{
    Positions positions;
    Velocities velocities;
    TrianglePositions trianglePositions[boids_size];
};

struct IDPair
{
    int id;
    int cellId;
};

//struct SortArrays
//{
//    int flags[boids_size];
//    int I_down[boids_size];
//    int I_up[boids_size];
//    int index[boids_size];
//};
//
//struct Indexes
//{
//    int j, bits, d;
//};


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

constexpr int   cellsInRow = (windowWidth / visual_range);
constexpr int   cellsInColumn = (windowHeight / visual_range);
constexpr int   cellsNum = cellsInRow * cellsInColumn;


constexpr unsigned countBits(unsigned int number)
{
    return (int)log2(number) + 1;
}

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

void Swap(IDPair* a, IDPair* b)
{
    IDPair t = *a;
    *a = *b;
    *b = t;
}

int Partition(IDPair* particlesGrid, int low, int high)
{
    int pivot = particlesGrid[high].cellId;
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (particlesGrid[j].cellId < pivot) {
            i++; // increment index of smaller element
            Swap(&particlesGrid[i], &particlesGrid[j]);
        }
    }
    Swap(&particlesGrid[i + 1], &particlesGrid[high]);
    return (i + 1);
}

void QuickSort(IDPair* particlesGrid, int low, int high)
{
    if (low < high) {
        int pi = Partition(particlesGrid, low, high);
        QuickSort(particlesGrid, low, pi - 1);
        QuickSort(particlesGrid, pi + 1, high);
    }
}

int* CalculateCellIDList(Boids* boids, int i, int &n)
{
    int neighborID[9];
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    int row = y / visual_range;
    int col = x / visual_range;
    int cellID = col + row * cellsInRow;

    neighborID[0] = cellID;
    n = 1;

    if(row-1>0 && col-1>0)    //left-top
    {
        neighborID[n] = cellID - cellsInRow - 1;
        n++;
    }
    if (row - 1 > 0)    //top
    {
        neighborID[n] = cellID - cellsInRow;
        n++;
    }
    if (row - 1 > 0 && col + 1 < cellsInRow)    //right-top
    {
        neighborID[n] = cellID - cellsInRow + 1;
        n++;
    }
    if (col - 1 > 0)    //left
    {
        neighborID[n] = cellID - 1;
        n++;
    }
    if (col + 1 < cellsInRow)    //right
    {
        neighborID[n] = cellID +1;
        n++;
    }
    if (row + 1 < cellsInColumn && col -1 > 0)    //left-bottom
    {
        neighborID[n] = cellID + cellsInRow - 1;
        n++;
    }
    if (row + 1 < cellsInColumn)    //bottom
    {
        neighborID[n] = cellID + cellsInRow;
        n++;
    }
    if (row + 1 < cellsInColumn && col + 1 < cellsInRow)    //right-bottom
    {
        neighborID[n] = cellID + cellsInRow + 1;
        n++;
    }

    return neighborID;
}

void RenderFrame(queue& q, Boids* boids, IDPair* particlesGrid, int* cellStart)
{
    // fill unordered list (id, cellid)
    range<1> num_items{ boids_size };

    q.submit([&](handler& h) {
        h.parallel_for(num_items, [=](id<1> i) {
            //int i = item.get_global_id();
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    //calc cell id
    int row = y / visual_range;
    int col = x / visual_range;
    int cellId = col + row * cellsInRow;
    particlesGrid[i].cellId = cellId;
    particlesGrid[i].id = i;
    cellStart[i] = -1;
    });
    });
    q.wait();

    QuickSort(particlesGrid, 0, boids_size - 1);

    //RADIX SORT
    //for (int i = 0; i < 1024; i++)
    //    tab[i] = 0;
    //for (int i = 0; i < boids_size; i++)
    //    tab[particlesGrid[i].id]++;

    //indexes->bits = 0;
    //indexes->j = 1;

    //int num_bits = countBits(cellsNum);
    //for (; indexes->bits < num_bits - 1; indexes->j *= 2, indexes->bits++)
    //{
    //    int j = indexes->j;
    //    q.parallel_for(num_items, [=](id<1> i) {
    //        int result = ((particlesGrid[i].cellId & indexes->j) == 0 ? 0 : 1);
    //    sortArrays->flags[i] = result;
    //    sortArrays->I_down[i] = !result;
    //    sortArrays->I_up[boids_size - i - 1] = result;
    //        }).wait();

    //        // I_down

    //        q.single_task([=]() {sortArrays->I_down[boids_size - 1] = 0; });
    //        for (indexes->d = log2(boids_size) - 1; indexes->d >= 0; indexes->d--)
    //        {
    //            int size = (boids_size - 1 / pow(2, indexes->d + 1));
    //            range<1> items{ size };
    //            q.parallel_for(items, [=](id<1> k) {
    //                int d = indexes->d;
    //            int t = sortArrays->I_down[k + (int)std::pow(2, d) - 1];
    //            sortArrays->I_down[k + (int)std::pow(2, d) - 1] = sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1];
    //            sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1] = t + sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1];
    //                }).wait();
    //        }

    //        // I_up
    //        for (indexes->d = 0; indexes->d <= log2(boids_size) - 1; indexes->d++)
    //        {
    //            int size = (boids_size - 1) / (pow(2, indexes->d + 1));
    //            range<1> items{ size };
    //            q.parallel_for(items, [=](id<1> k) {
    //                int d = indexes->d;
    //            sortArrays->I_up[k + (int)std::pow(2, d + 1) - 1] = sortArrays->I_up[k + (int)std::pow(2, d) - 1] + sortArrays->I_up[k + (int)std::pow(2, d + 1) - 1];
    //                }).wait();
    //        }


    //        q.parallel_for(num_items, [=](id<1> i) {
    //            sortArrays->I_up[i] = boids_size - sortArrays->I_up[i];
    //        particlesGridHelper[i].cellId = particlesGrid[i].cellId;
    //        particlesGridHelper[i].id = particlesGrid[i].id;
    //        if (sortArrays->flags[i])
    //            sortArrays->index[i] = sortArrays->I_up[i];
    //        else
    //            sortArrays->index[i] = sortArrays->I_down[i];
    //            }).wait();

    //            q.parallel_for(num_items, [=](id<1> i) {
    //                int index = sortArrays->index[i];
    //            particlesGrid[index].cellId = particlesGridHelper[i].cellId;
    //            particlesGrid[index].id = particlesGridHelper[i].id;
    //                }).wait();
    //}

    range<1> num_items_reduced{ boids_size - 1 };

    q.single_task([=]() {cellStart[particlesGrid[0].cellId] = 0; });
    q.parallel_for(num_items_reduced, [=](id<1> i) {
        if (particlesGrid[i + 1].cellId != particlesGrid[i].cellId)
            cellStart[particlesGrid[i].cellId] = i;
        }).wait();

    q.parallel_for(num_items, [=](id<1> i) {
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    float vx = boids->velocities.vx[i];
    float vy = boids->velocities.vy[i];
    auto distance = [&](float x1, float y1, float x2, float y2)
    {
        return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    };
    // NEW VELOCITY
    float close_dx = 0.0f;
    float close_dy = 0.0f;

    float vx_avg = 0.0f;
    float vy_avg = 0.0f;

    float x_avg = 0.0f;
    float y_avg = 0.0f;

    unsigned int neighbors = 0;

    // process every neighbor cell
    int n;
    int* neighborCells = CalculateCellIDList(boids, i, n);

    for (int counter = 0; counter < n; counter++)
    {
        int cellNum = neighborCells[counter];
        int particleNum = cellStart[cellNum];
        while (particlesGrid[particleNum].cellId == cellNum)
        {
            int j = particlesGrid[particleNum].id;
            particleNum++;
            if (j == i) continue;
            float friend_x = boids->positions.x[j];
            float friend_y = boids->positions.y[j];
            float dist = distance(x, y, friend_x, friend_y);
            if (dist > visual_range)
                continue;

            float friend_vx = boids->velocities.vx[j];
            float friend_vy = boids->velocities.vy[j];

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
    float vp_x = -boids->velocities.vy[i];
    float vp_y = boids->velocities.vx[i];
    float scale = 2 / sqrt(vp_x * vp_x + vp_y * vp_y);
    vp_x *= scale;
    vp_y *= scale;

    // update velocity
    boids->velocities.vx[i] = vx;
    boids->velocities.vy[i] = vy;

    // update position
    float new_x = x + vx;
    float new_y = y + vy;
    boids->positions.x[i] = new_x;
    boids->positions.y[i] = new_y;


    boids->trianglePositions[i].p1_x = new_x + vp_x;
    boids->trianglePositions[i].p1_y = new_y + vp_y;
    boids->trianglePositions[i].p2_x = new_x - vp_x;
    boids->trianglePositions[i].p2_y = new_y - vp_y;
    boids->trianglePositions[i].p3_x = new_x + vx * scale * 2.5;
    boids->trianglePositions[i].p3_y = new_y + vy * scale * 2.5;
        });
}

void InitializeInput(Boids &boids)
{
    srand(time(NULL));
    for (size_t i = 0; i < boids_size; i++)
    {
        boids.positions.x[i] = rand() % static_cast<int>(windowWidth - margin * 2) + margin;
        boids.positions.y[i] = rand() % static_cast<int>(windowHeight - margin * 2) + margin;
        float randAngle = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 3.141592653589 * 2;
        boids.velocities.vx[i] = minspeed * cos(randAngle);
        boids.velocities.vy[i] = minspeed * sin(randAngle);
        float vp_x = -boids.velocities.vy[i];
        float vp_y = boids.velocities.vx[i];
        float scale = 2 / sqrt(vp_x * vp_x + vp_y * vp_y);
        vp_x *= scale;
        vp_y *= scale;
        
        boids.trianglePositions[i].p1_x = boids.positions.x[i] + vp_x;
        boids.trianglePositions[i].p1_y = boids.positions.y[i] + vp_y;
        boids.trianglePositions[i].p2_x = boids.positions.x[i] - vp_x;
        boids.trianglePositions[i].p2_y = boids.positions.y[i] - vp_y;
        boids.trianglePositions[i].p3_x = boids.positions.x[i] + boids.velocities.vx[i] * scale * 2.5;
        boids.trianglePositions[i].p3_y = boids.positions.y[i] + boids.velocities.vy[i] * scale * 2.5;
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

    Boids cpu_boids;

    //CPUBoids cpu_boids;
    InitializeInput(cpu_boids);

    // Copy all buffers to GPU memory
    IDPair* particlesGrid = (IDPair*)malloc_shared(boids_size * sizeof(IDPair), q);
    Boids* gpu_boids = (Boids*)malloc_device(sizeof(Boids), q);
    int* cellStart = (int*)malloc_device(boids_size* sizeof(int), q);

    //Indexes* indexes = (Indexes*)malloc_shared(sizeof(Indexes), q);
    //IDPair* particlesGridHelper = (IDPair*)malloc_shared(boids_size * sizeof(IDPair), q);
     //SortArrays* sortArrays = (SortArrays*)malloc_shared(sizeof(SortArrays), q);

    q.submit([&](handler& h) {
        h.memcpy(gpu_boids, &cpu_boids, sizeof(Boids));}).wait();

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
            //printf("%f ms/frame\n", 1000.0 / double(nbFrames));
            printf("%d FPS\n", nbFrames);
            nbFrames = 0;
            lastTime += 1.0;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        //schedule frame to render
        RenderFrame(q, gpu_boids, particlesGrid, cellStart);
        //draw
        glBufferData(GL_ARRAY_BUFFER, boids_size * sizeof(float)*6, &(cpu_boids.trianglePositions), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, boids_size * 3);
        glfwSwapBuffers(window);
        glfwPollEvents();

        //wait until frame is rendered and copy rendered frame to host
        q.wait();
        q.submit([&](handler& h) {
            h.memcpy(&(cpu_boids.trianglePositions), &(gpu_boids->trianglePositions),boids_size * sizeof(float)*6);}).wait();
    }
    glfwTerminate();

    free(gpu_boids, q);
    free(particlesGrid, q);
    free(cellStart);
    //free(sortArrays, q);
   // free(indexes, q);
    return 0;
}