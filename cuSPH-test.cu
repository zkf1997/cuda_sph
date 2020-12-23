#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>

#include "cuSPH.h"

using namespace std;

class cuSPH3D
{
public:
    struct Box box = Box(1.0, 1.0, 1.5);
    cuSPH3DMap* map;
    cuSPHParticles* particles;
    int m_num_p = 12000;
    float m_k = 1000;
    float m_h = 0.065;
    float m_dt = 0.4 * m_h / sqrt(m_k);
    float m_rho0 = 1000;
    float m_mass = pow((m_h / 2), 3) * m_rho0;
    float m_gamma = 1;
    float m_mu = 0.05;
    int m_fext = 0;
    int m_ctrl = 0;
    float m_time = 0.0;
    int m_step = 0;
    int num_cells;
    float cell_size;
    int max_per_cell;

    cuSPH3D()
    {
        m_num_p = 1024 * 16;
        map = new cuSPH3DMap(box, m_h);
        cout << "map contructed" << endl;
        num_cells = map->num_cells;
        cell_size = m_h;
        max_per_cell = map->k;
        cout << "num_cells:" << num_cells << " cell_size:" << cell_size << "max particles in one cell:" << max_per_cell << endl;
        particles = new cuSPHParticles(m_num_p);
        cout << "created " << m_num_p << " particles" << endl;
        cudaDeviceSynchronize();
        init();
        cout << "init finished" << endl;
    }

    virtual ~cuSPH3D()
    {
    }

public:
    void init()
    {
        std::unique_ptr<Particle[]>& p = particles->getHostPtr();
        auto size = m_num_p;

        float d = m_h / 2;
        int placed = 0;
        float xpos = box.xmin + d;
        float ypos = box.ymin + d;
        float zpos = 10 * d;

        while (placed < m_num_p) {
            if (xpos >= box.xmax / 1.0) {
                xpos = box.xmin + d;
                ypos = ypos + d;
                if (ypos >= box.ymax) {
                    ypos = box.ymin + d;
                    zpos = zpos - d;
                }
            }
            
            //cout << "point:" << placed << endl;
            p[placed].pos = glm::vec3(xpos, ypos, zpos);
            xpos = xpos + d;
            placed = placed + 1;
        }

        for (int i = 0; i < size; i++)
        {
            //Position
            /*p[i].pos.x = 10.0 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            p[i].pos.y = 10.0 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            p[i].pos.z = 10.0 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);*/

            //Velocity
            p[i].vel.x = 0;
            p[i].vel.y = 0;
            p[i].vel.z = 0;

            p[i].r = 0.5;
            p[i].g = 0.5;
            p[i].b = 0;

            p[i].mass = m_mass;
            p[i].radius = 0.1;
        }
        //Set to CUDA device
        particles->setToDevice();
    }

    void sort()
    {
        //cout << "reset cells" << endl;
        cellsReset <<<(num_cells + 511)/512, 512>>>
            (map->cells_device, num_cells);
        //cout << "reset finish" << endl;
        cellsSort<Particle>
            <<<(m_num_p + 511) / 512, 512>>>
            (particles->particles_device, m_num_p, map->cells_device, m_h, map->box_device, max_per_cell);
        //cout << "sort finish" << endl;
        cudaDeviceSynchronize();
    }

    void updateStates() 
    {
        //cout << "update states" << endl;
        update_states_kernel <<<(m_num_p + 511) / 512, 512 >>>
            (particles->particles_device, m_num_p, map->cells_device
                , map->box_device, m_h, m_rho0, m_gamma, m_k);
        cudaDeviceSynchronize();

    }

    void computeForce() 
    {
        //cout << "update forces" << endl;
        compute_force_kernel <<<(m_num_p + 511) / 512, 512>>> 
            (particles->particles_device, m_num_p, map->cells_device
            , map->box_device, m_h, m_mu, m_fext);
        auto error = cudaGetLastError();
        if (error != 0)
            cout << error << endl;
        cudaDeviceSynchronize();

    }

    void advection() 
    {
        //cout << "advection" << endl;
        advection_kernel <<<(m_num_p + 511) / 512, 512>>> (particles->particles_device, m_num_p, m_dt);
        cudaDeviceSynchronize();
    }

    void advance()
    {
        sort();
        updateStates();
        computeForce();
        advection();

        m_time += m_dt;
        m_step++;
    }
};

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.5f;
const float ZOOM = 45.0f;


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }
    // constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
        Position = glm::vec3(posX, posY, posZ);
        WorldUp = glm::vec3(upX, upY, upZ);
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
        return glm::lookAt(Position, Position + Front, Up);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == DOWN)
            Position -= Up * velocity;
        if (direction == UP)
            Position += Up * velocity;
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        Zoom -= (float)yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(10.0f, 5.0f, 10.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

void display(GLFWwindow* window, cuSPH3D* sph)
{
    // per-frame time logic
        // --------------------
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    // -----
    processInput(window);

    glClearColor(0, 0.2, 0.2, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /*int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, w, h);*/

    // create a world with dimensions x:[-SIM_W,SIM_W] and y:[0,SIM_W*2]
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-20, 20, -20, 20, -20, 10);

    /*glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();*/

    //cout << "retrieve particles" << endl;
    auto& particles = sph->particles->getFromDevice();
    vector<glm::vec3> particles_transformed(sph->m_num_p);
    cout << "got particles" << endl;
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = camera.GetViewMatrix();
    for (int i = 0; i < sph->m_num_p; i++)
    {
        /*cout << particles[i].pos.x << ' ' << particles[i].pos.y << ' ' << particles[i].pos.z << ' ' <<  particles[i].r << endl;*/
        particles_transformed[i] = glm::vec3(projection * view * glm::vec4(particles[i].pos, 1.0f));
        //particles_transformed[i] = glm::vec3(projection * view * glm::vec4(glm::vec3(float(i) / sph->m_num_p, float(i) / sph->m_num_p, float(i) / sph->m_num_p), 1.0f));
        //cout << particles_transformed[i].x << ' ' << particles_transformed[i].y << ' ' << particles_transformed[i].z << ' ' << endl;
    }

    // Draw Fluid Particles
    glPointSize(2);
    //cout << 1 << endl;
    glVertexPointer(3, GL_FLOAT, sizeof(glm::vec3), &(particles_transformed[0]));
    glColorPointer(3, GL_FLOAT, sizeof(Particle), &(particles[0].r));
    //cout << 2 << endl;
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    //cout << 3 << endl;
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(sph->m_num_p));
    //cout << 4 << endl;
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    //cout << 5 << endl;
    
    //draw axis
    glm::vec3 x, y, z, o;
    o = glm::vec3(projection * view * glm::vec4(glm::vec3(0.f, 0.f, 0.f), 1.0f));
    x = glm::vec3(projection * view * glm::vec4(glm::vec3(1.f, 0.f, 0.f), 1.0f));
    y = glm::vec3(projection * view * glm::vec4(glm::vec3(0.f, 1.f, 0.f), 1.0f));
    z = glm::vec3(projection * view * glm::vec4(glm::vec3(0.f, 0.f, 1.f), 1.0f));
    glPointSize(10);
    glLineWidth(2.5);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(o.x, o.y, o.z);
    glVertex3f(x.x, x.y, x.z);
    glEnd();
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(o.x, o.y, o.z);
    glVertex3f(y.x, y.y, y.z);
    glEnd();
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(o.x, o.y, o.z);
    glVertex3f(z.x, z.y, z.z);
    glEnd();
}

int main()
{
    cuSPH3D sph;
    //cout << 1 << endl;
    glfwInit();
    //cout << 2 << endl;
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SPH", NULL, NULL);
    if (window == NULL) {
        cout << "Failed to create GLFW window!" << endl;
        glfwTerminate();
        std::exit(-1);
    }
    //cout << 3 << endl;
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    //cout << 4 << endl;
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    //cout << 5 << endl;

    glfwSwapInterval(1);

    //glfwSetKeyCallback(window, keyboard);
    //glfwSetCursorPosCallback(window, motion);
    //glfwSetMouseButtonCallback(window, mouse);
    
    
    while (!glfwWindowShouldClose(window))
    {
        cout << "to display frame" << endl;
        display(window, &sph);

        glfwPollEvents();

        sph.advance();

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}