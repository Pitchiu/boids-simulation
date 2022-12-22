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

#include <windows.h>

#include "constants.h"
#include "shaders.h"

#define __cdecl
#define __stdcall

using namespace sycl;


struct Positions
{
    float x[kUnitCount];
    float y[kUnitCount];
};

struct Velocities
{
    float vx[kUnitCount];
    float vy[kUnitCount];
};

struct Point
{
    float x;
    float y;
};

struct TrianglePositions
{
    Point p1, p2, p3;
};

struct Boids
{
    Positions positions;
    Velocities velocities;
    TrianglePositions trianglePositions[kUnitCount];
};

struct IdPair
{
    int id;
    int cellId;
};

bool pauseFlag = false;


static auto ExceptionHandler = [](sycl::exception_list e_list) {
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

void PauseCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        pauseFlag = !pauseFlag;
}

void Swap(IdPair* a, IdPair* b)
{
    IdPair t = *a;
    *a = *b;
    *b = t;
}

int Partition(IdPair* particlesGrid, int low, int high)
{
    int pivot = particlesGrid[high].cellId;
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (particlesGrid[j].cellId < pivot) {
            i++;
            Swap(&particlesGrid[i], &particlesGrid[j]);
        }
    }
    Swap(&particlesGrid[i + 1], &particlesGrid[high]);
    return (i + 1);
}

void QuickSort(IdPair* particlesGrid, int low, int high)
{
    if (low < high) {
        int pi = Partition(particlesGrid, low, high);
        QuickSort(particlesGrid, low, pi - 1);
        QuickSort(particlesGrid, pi + 1, high);
    }
}

int* CalculateCellIdList(Boids* boids, int i, int &n)
{
    int neighborID[9];
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    int row = y / kVisualRange;
    int col = x / kVisualRange;
    int cellID = col + row * kGridColsNum;

    neighborID[0] = cellID;
    n = 1;

    if(row-1>0 && col-1>0)    // left-top
    {
        neighborID[n] = cellID - kGridColsNum - 1;
        n++;
    }
    if (row - 1 > 0)    // top
    {
        neighborID[n] = cellID - kGridColsNum;
        n++;
    }
    if (row - 1 > 0 && col + 1 < kGridColsNum)    // right-top
    {
        neighborID[n] = cellID - kGridColsNum + 1;
        n++;
    }
    if (col - 1 > 0)    // left
    {
        neighborID[n] = cellID - 1;
        n++;
    }
    if (col + 1 < kGridColsNum)    // right
    {
        neighborID[n] = cellID +1;
        n++;
    }
    if (row + 1 < kGridRowsNum && col -1 > 0)    // left-bottom
    {
        neighborID[n] = cellID + kGridColsNum - 1;
        n++;
    }
    if (row + 1 < kGridRowsNum)    // bottom
    {
        neighborID[n] = cellID + kGridColsNum;
        n++;
    }
    if (row + 1 < kGridRowsNum && col + 1 < kGridColsNum)    // right-bottom
    {
        neighborID[n] = cellID + kGridColsNum + 1;
        n++;
    }

    return neighborID;
}

void RenderFrame(queue& q, Boids* boids, IdPair* particlesGrid, int* cellStart, Positions* temporaryPositions, Velocities* temporaryVelocities, Point* mousePointer)
{
    // Fill unordered list (id, cellid)
    range<1> numItems{ kUnitCount };

    q.submit([&](handler& h) {
        h.parallel_for(numItems, [=](id<1> i) {
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    int row = y / kVisualRange;
    int col = x / kVisualRange;
    int cellId = col + row * kGridColsNum;
    particlesGrid[i].cellId = cellId;
    particlesGrid[i].id = i;
    cellStart[i] = -1;
    });
    });
    q.wait();

    // Sort the list with quicksort
    QuickSort(particlesGrid, 0, kUnitCount - 1);

    //RADIX SORT
    //for (int i = 0; i < 1024; i++)
    //    tab[i] = 0;
    //for (int i = 0; i < kUnitCount; i++)
    //    tab[particlesGrid[i].id]++;

    //indexes->bits = 0;
    //indexes->j = 1;

    //int num_bits = CountBits(kCellsNumTotal);
    //for (; indexes->bits < num_bits - 1; indexes->j *= 2, indexes->bits++)
    //{
    //    int j = indexes->j;
    //    q.parallel_for(numItems, [=](id<1> i) {
    //        int result = ((particlesGrid[i].cellId & indexes->j) == 0 ? 0 : 1);
    //    sortArrays->flags[i] = result;
    //    sortArrays->I_down[i] = !result;
    //    sortArrays->I_up[kUnitCount - i - 1] = result;
    //        }).wait();

    //        // I_down

    //        q.single_task([=]() {sortArrays->I_down[kUnitCount - 1] = 0; });
    //        for (indexes->d = log2(kUnitCount) - 1; indexes->d >= 0; indexes->d--)
    //        {
    //            int size = (kUnitCount - 1 / pow(2, indexes->d + 1));
    //            range<1> items{ size };
    //            q.parallel_for(items, [=](id<1> k) {
    //                int d = indexes->d;
    //            int t = sortArrays->I_down[k + (int)std::pow(2, d) - 1];
    //            sortArrays->I_down[k + (int)std::pow(2, d) - 1] = sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1];
    //            sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1] = t + sortArrays->I_down[k + (int)std::pow(2, d + 1) - 1];
    //                }).wait();
    //        }

    //        // I_up
    //        for (indexes->d = 0; indexes->d <= log2(kUnitCount) - 1; indexes->d++)
    //        {
    //            int size = (kUnitCount - 1) / (pow(2, indexes->d + 1));
    //            range<1> items{ size };
    //            q.parallel_for(items, [=](id<1> k) {
    //                int d = indexes->d;
    //            sortArrays->I_up[k + (int)std::pow(2, d + 1) - 1] = sortArrays->I_up[k + (int)std::pow(2, d) - 1] + sortArrays->I_up[k + (int)std::pow(2, d + 1) - 1];
    //                }).wait();
    //        }


    //        q.parallel_for(numItems, [=](id<1> i) {
    //            sortArrays->I_up[i] = kUnitCount - sortArrays->I_up[i];
    //        particlesGridHelper[i].cellId = particlesGrid[i].cellId;
    //        particlesGridHelper[i].id = particlesGrid[i].id;
    //        if (sortArrays->flags[i])
    //            sortArrays->index[i] = sortArrays->I_up[i];
    //        else
    //            sortArrays->index[i] = sortArrays->I_down[i];
    //            }).wait();

    //            q.parallel_for(numItems, [=](id<1> i) {
    //                int index = sortArrays->index[i];
    //            particlesGrid[index].cellId = particlesGridHelper[i].cellId;
    //            particlesGrid[index].id = particlesGridHelper[i].id;
    //                }).wait();
    //}

    range<1> numItemsReduced{ kUnitCount - 1 };

    // Fill cellStart array
    q.single_task([=]() {cellStart[particlesGrid[0].cellId] = 0; });
    q.parallel_for(numItemsReduced, [=](id<1> i) {
        if (particlesGrid[i + 1].cellId != particlesGrid[i].cellId)
            cellStart[particlesGrid[i].cellId] = i;
        }).wait();

    q.parallel_for(numItems, [=](id<1> i) {
    float x = boids->positions.x[i];
    float y = boids->positions.y[i];
    float vx = boids->velocities.vx[i];
    float vy = boids->velocities.vy[i];
    auto distance = [&](float x1, float y1, float x2, float y2)
    {
        return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    };
    // New velocity
    float xAvoid = 0.0f;
    float yAvoid = 0.0f;

    float vxAvg = 0.0f;
    float vyAvg = 0.0f;

    float xAvg = 0.0f;
    float yAvg = 0.0f;

    unsigned int neighbors = 0;

    // Process every neighbor cell
    int n;
    int* neighborCells = CalculateCellIdList(boids, i, n);

    for (int counter = 0; counter < n; counter++)
    {
        int cellNum = neighborCells[counter];
        int particleNum = cellStart[cellNum];
        while (particlesGrid[particleNum].cellId == cellNum)
        {
            int j = particlesGrid[particleNum].id;
            particleNum++;
            if (j == i) continue;
            float xFriend = boids->positions.x[j];
            float yFriend = boids->positions.y[j];
            float dist = distance(x, y, xFriend, yFriend);
            if (dist > kVisualRange)
                continue;

            float xFriendVelocity = boids->velocities.vx[j];
            float yFriendVelocity = boids->velocities.vy[j];

            if (dist < kProtectedRange)
            {
                xAvoid += x - xFriend;
                yAvoid += y - yFriend;
                continue;
            }
            neighbors++;
            vxAvg += xFriendVelocity;
            vyAvg += yFriendVelocity;
            xAvg += xFriend;
            yAvg += yFriend;

        }
    }
    // Mouse pointer avoiding
    int xMouse = mousePointer->x;
    int yMouse = mousePointer->y;
    int xMouseAvoid = 0;
    int yMouseAvoid = 0;
    float pointerDistance = distance(x, y, xMouse, yMouse);

    if (pointerDistance < kVisualRange)
    {
        xMouseAvoid = x - xMouse;
        yMouseAvoid = y - yMouse;
    }
    vx += xMouseAvoid * kMouseFactor;
    vy += yMouseAvoid * kMouseFactor;

    if (neighbors > 0)
    {
        // Alignment
        vxAvg /= neighbors;
        vyAvg /= neighbors;
        vx += (vxAvg - vx) * kAlignFactor;
        vy += (vyAvg - vy) * kAlignFactor;

        // Cohesion
        xAvg /= neighbors;
        yAvg /= neighbors;
        vx += (xAvg - x) * kCenteringFactor;
        vy += (yAvg - y) * kCenteringFactor;
    }

    // Separation
    vx += xAvoid * kAvoidFactor;
    vy += yAvoid * kAvoidFactor;


    // Margin
    if (x < kLeftMarginSize)
        vx += kTurnFactor;
    else if (x > kRightMarginSize)
        vx -= kTurnFactor;
    if (y < kBottomMarginSize)
        vy += kTurnFactor;
    else if (y > kTopMarginSize)
        vy -= kTurnFactor;

    // Speed limit
    float speed = sqrt(vx * vx + vy * vy);
    if (speed > kMaxSpeed)
    {
        vx = vx / speed * kMaxSpeed;
        vy = vy / speed * kMaxSpeed;
    }

    if (speed < kMinSpeed)
    {
        vx = vx / speed * kMinSpeed;
        vy = vy / speed * kMinSpeed;
    }

    float xVelocity = -boids->velocities.vy[i];
    float yVelocity = boids->velocities.vx[i];
    float scale = 2 / sqrt(xVelocity * xVelocity + yVelocity * yVelocity);
    xVelocity *= scale;
    yVelocity *= scale;

    float xNew = x + vx;
    float yNew = y + vy;

    // Write velocity and position to temporary buffer
    temporaryVelocities->vx[i] = vx;
    temporaryVelocities->vy[i] = vy;
    temporaryPositions->x[i] = xNew;
    temporaryPositions->y[i] = yNew;

    // Triangle positions can be updated safely
    boids->trianglePositions[i].p1.x = xNew + xVelocity;
    boids->trianglePositions[i].p1.y = yNew + yVelocity;
    boids->trianglePositions[i].p2.x = xNew - xVelocity;
    boids->trianglePositions[i].p2.y = yNew - yVelocity;
    boids->trianglePositions[i].p3.x = xNew + vx * scale * 2.5;
    boids->trianglePositions[i].p3.y = yNew + vy * scale * 2.5;
        }).wait();

    // Update position
    q.parallel_for(numItems, [=](id<1> i) {
    boids->positions.x[i] = temporaryPositions->x[i];
    boids->positions.y[i] = temporaryPositions->y[i];
    boids->velocities.vx[i] = temporaryVelocities->vx[i];
    boids->velocities.vy[i] = temporaryVelocities->vy[i];
        }).wait();
}

void InitializeInput(Boids &boids)
{
    srand(time(NULL));
    for (size_t i = 0; i < kUnitCount; i++)
    {
        boids.positions.x[i] = rand() % static_cast<int>(kWindowWidth - kMarginSize * 2) + kMarginSize;
        boids.positions.y[i] = rand() % static_cast<int>(kWindowHeight - kMarginSize * 2) + kMarginSize;
        float randAngle = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 3.141592653589 * 2;
        boids.velocities.vx[i] = kMinSpeed * cos(randAngle);
        boids.velocities.vy[i] = kMinSpeed * sin(randAngle);
        float xVelocity = -boids.velocities.vy[i];
        float yVelocity = boids.velocities.vx[i];
        float scale = 2 / sqrt(xVelocity * xVelocity + yVelocity * yVelocity);
        xVelocity *= scale;
        yVelocity *= scale;
        
        boids.trianglePositions[i].p1.x = boids.positions.x[i] + xVelocity;
        boids.trianglePositions[i].p1.y = boids.positions.y[i] + yVelocity;
        boids.trianglePositions[i].p2.x = boids.positions.x[i] - xVelocity;
        boids.trianglePositions[i].p2.y = boids.positions.y[i] - yVelocity;
        boids.trianglePositions[i].p3.x = boids.positions.x[i] + boids.velocities.vx[i] * scale * 2.5;
        boids.trianglePositions[i].p3.y = boids.positions.y[i] + boids.velocities.vy[i] * scale * 2.5;
    }
}


int main(int argc, char* argv[]) {
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, PauseCallback);
    if (glewInit() != GLEW_OK)
        std::cout << "Error";

    unsigned int buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    default_selector defaultSelector;
    queue q(defaultSelector, ExceptionHandler);

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    glm::mat4 proj = glm::ortho(0.0f, (float)kWindowWidth, 0.0f, (float)kWindowHeight, -1.0f, 1.0f);
    int location = glGetUniformLocation(shader, "u_MVP");
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(proj));

    Boids cpuBoids;
    InitializeInput(cpuBoids);

    // Allocate and fill buffers in GPU memory
    IdPair* particlesGrid = (IdPair*)malloc_shared(kUnitCount * sizeof(IdPair), q);
    Boids* gpuBoids = (Boids*)malloc_device(sizeof(Boids), q);
    Positions* temporaryPositions = (Positions*)malloc_device(sizeof(Positions), q);
    Velocities* temporaryVelocities = (Velocities*)malloc_device(sizeof(Velocities), q);
    Point *mousePointer = (Point*)malloc_shared(sizeof(Point), q);

    int* cellStart = (int*)malloc_device(kUnitCount* sizeof(int), q);

    q.submit([&](handler& h) {
        h.memcpy(gpuBoids, &cpuBoids, sizeof(Boids));}).wait();

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while (!glfwWindowShouldClose(window))
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        mousePointer->x = xpos;
        mousePointer->y = kWindowHeight - ypos;

        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            printf("%d FPS\n", nbFrames);
            nbFrames = 0;
            lastTime += 1.0;
        }
        while (pauseFlag)
        {
            Sleep(100);
            glfwPollEvents();
        }

        glClear(GL_COLOR_BUFFER_BIT);
        // schedule frame to render
        RenderFrame(q, gpuBoids, particlesGrid, cellStart, temporaryPositions, temporaryVelocities, mousePointer);
        //draw
        glBufferData(GL_ARRAY_BUFFER, kUnitCount * sizeof(float)*6, &(cpuBoids.trianglePositions), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, kUnitCount * 3);
        glfwSwapBuffers(window);
        glfwPollEvents();

        //wait until frame is rendered and copy rendered frame to host
        q.submit([&](handler& h) {
            h.memcpy(&(cpuBoids.trianglePositions), &(gpuBoids->trianglePositions),kUnitCount * sizeof(float)*6);}).wait();
    }
    glfwTerminate();

    free(gpuBoids, q);
    free(particlesGrid, q);
    free(temporaryPositions, q);
    free(temporaryVelocities, q);
    free(cellStart, q);
    free(mousePointer, q);
    return 0;
}