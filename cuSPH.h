#pragma once

#include <type_traits>

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

struct Cell
{
	unsigned int count;
	unsigned int* list;
};

struct Box
{
	Box()
	{
		xmin = ymin = zmin = 0;
		xmax = 1.0;
		ymax = 1.0;
		zmax = 1.5;
		xres = yres = zres = 64;
		rmin = 0.1;
	}

	Box(float xmax, float ymax, float zmax):xmax(xmax), ymax(ymax), zmax(zmax)
	{
		xmin = ymin = zmin = 0;
		xres = yres = zres = 64;
		rmin = 0.1;
	}

	float xmin, xmax, xres;// min and max bound along x, and the num of space partitions along x
	float ymin, ymax, yres;
	float zmin, zmax, zres;

	float rmin;
};

struct Particle
{
	glm::vec3 pos;
	glm::vec3 vel;
	glm::vec3 force;

	float radius;
	float mass;
	float density = 0;
	float pressure = 0;

	//cell index 
	glm::ivec3 cell_idx;
	//idx of this particle in its cell
	int no_in_cell;

	//for rendering
	float r, g, b;
};

struct ParticleTmp
{
	glm::vec3 pos;
	glm::vec3 vel;
};

class cuSPHParticles
{
	public:
		Particle* particles_device;
		ParticleTmp* particles_device_tmp;
		std::unique_ptr<Particle[]> particles_host;
		unsigned int num_particles;

		unsigned int num_ctrl_particles;
		ParticleTmp* ctrl_particles_device;

		cuSPHParticles() {}

		cuSPHParticles(int n):particles_host(new Particle[n])
		{
			cout << n << endl;
			num_particles = n;
			//particles_host = (Particle*)calloc(n, sizeof(Particle));
			/*if (particles_host == NULL)
				cout << "alloc fail" << endl;*/
			cudaMalloc(
					reinterpret_cast<void**>(&particles_device),
					n*sizeof(Particle)
					);
			/*cudaMalloc(
					reinterpret_cast<void**>(&particles_device_tmp),
					n*sizeof(ParticleTmp)
					);*/
		}

		~cuSPHParticles()
		{
			cudaFree(reinterpret_cast<void*>(particles_device));
			//cudaFree(reinterpret_cast<void*>(particles_device_tmp));
			//delete(particles_host);
		}

		std::unique_ptr<Particle[]>& getFromDevice(void)
		{
			cudaMemcpy(
					particles_host.get(),
					particles_device,
					num_particles *sizeof(Particle),
					cudaMemcpyDeviceToHost
					);
			return particles_host;
		}

		void setToDevice(void)
		{
			cudaMemcpy(
					particles_device,
					particles_host.get(),
					num_particles *sizeof(Particle),
					cudaMemcpyHostToDevice
					);
		}

		std::unique_ptr<Particle[]>& getHostPtr(void)
		{
			return particles_host;
		}
};



__global__ void cellsAlloc(struct Cell *cells_device, unsigned int k, int num_cells)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (id < num_cells)
	{
		cells_device[id].count = 0U;
		cells_device[id].list
			= reinterpret_cast<unsigned int*>(malloc(k * sizeof(unsigned int)));
	}
}

__global__ void cellsFree(struct Cell* cells_device, int num_cells)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < num_cells)
		free(reinterpret_cast<void*>(cells_device[id].list));
}

__global__ void cellsReset(struct Cell* cells_device, int num_cells)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < num_cells)
		cells_device[id].count = 0U;
}

//sort for particle or control particle
template <typename A>
__global__ static void cellsSort(A* particles_device, int num_particles, struct Cell *cells_device, float cell_size, struct Box* box_device, int k)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= num_particles)
		return;

	A& particle = particles_device[id];

	auto px = particle.pos.x,
			py = particle.pos.y,
			pz = particle.pos.z;

	auto xmax = box_device->xmax,
			ymax = box_device->ymax,
			zmax = box_device->zmax;
	auto xmin = box_device->xmin,
		ymin = box_device->ymin,
		zmin = box_device->zmin;
	auto xres = box_device->xres,
		yres = box_device->yres,
		zres = box_device->zres;

	/*auto dx = xmax/(xres-2),
			dy = ymax/(yres-2),
			dz = zmax/(zres-2);*/
	float dx, dy, dz;
	dx = dy = dz = cell_size;

	unsigned int x,y,z;

	if(px<xmin)
	{
		x=0U;
	}
	else if(px>xmax)
	{
		x=xres-1U;
	}
	else
	{
		x=static_cast<unsigned int>((px - xmin)/dx);
	}

	if(py<ymin)
	{
		y=0U;
	}
	else if(py>ymax)
	{
		y=yres-1U;
	}
	else
	{
		y=static_cast<unsigned int>((py - ymin)/dy);
	}

	if(pz<zmin)
	{
		z=0U;
	}
	else if(pz>zmax)
	{
		z=zres-1U;
	}
	else
	{
		z=static_cast<unsigned int>((pz - zmin)/dz);
	}

	unsigned int cell_idx = x * yres * zres + y * zres + z;
	particle.cell_idx = glm::ivec3(x, y, z);
	particle.no_in_cell = atomicAdd(&cells_device[cell_idx].count,1U);

	// printf("%u, %u, %u, %u\n",particle.bucket_loc.no,x,y,z);
	if (particle.no_in_cell < k)
		cells_device[cell_idx].list[particle.no_in_cell]=id;
}

class cuSPH3DMap
{
	public:
		struct Box box_host;
		struct Box* box_device;
		struct Cell* cells_device;
		int num_cells;
		int X, Y, Z;
		float cell_size;
		int k;

		cuSPH3DMap() {}

		cuSPH3DMap(struct Box box, float cell_size):cell_size(cell_size)
		{
			box_host = box;
			X = ceil((box.xmax - box.xmin) / cell_size);
			Y = ceil((box.ymax - box.ymin) / cell_size);
			Z = ceil((box.zmax - box.zmin) / cell_size);
			box_host.xres = X;
			box_host.yres = Y;
			box_host.zres = Z;
			num_cells = X * Y * Z;

			cudaMalloc(
					reinterpret_cast<void**>(&box_device),
					sizeof(struct Box)
					);
			cudaMemcpy(
					reinterpret_cast<void*>(box_device),
					reinterpret_cast<void*>(&box_host),
					sizeof(struct Box),
					cudaMemcpyHostToDevice
					);

			cudaMalloc(
					reinterpret_cast<void**>(&cells_device),
					X*Y*Z*sizeof(struct Cell)
					);

		    float dx = cell_size; 
			float dy = cell_size;
			float dz = cell_size;

			auto vf = dx * dy * dz * 0.8;
			auto vb = 4.0 * M_PI * std::pow(box.rmin, 3.0) / 3.0;
			k = vf/vb;
			k++;
			k = max(k, 128);
			cellsAlloc<<<(num_cells + 511) / 512, 512>>>(cells_device, k, num_cells);
		}

		~cuSPH3DMap()
		{
			cudaFree(reinterpret_cast<void*>(box_device));
			cudaFree(reinterpret_cast<void*>(cells_device));
			cellsFree<<<(num_cells + 511) / 512, 512 >>>(cells_device, num_cells);
		}
};

__host__ __device__ float W(glm::vec3 diff, float m_h) {
	float r = glm::length(diff);
	float q = r / m_h;

	float sigma_d = 8.0 / (M_PI * m_h * m_h * m_h);
	if (q <= 1.0)
	{
		if (q <= 0.5)
		{
			float q2 = q * q;
			float q3 = q2 * q;
			return (sigma_d * (6 * (q3 - q2) + 1));
		}
		else
		{
			return (sigma_d * 2 * (1 - q) * (1 - q) * (1 - q));
		}
	}

	return 0;
}

__host__ __device__ glm::vec3 gradW(glm::vec3 diff, float m_h) {
	float r = glm::length(diff);
	float q = r / m_h;

	float sigma_d = 48.0 / (M_PI * m_h * m_h * m_h);
	if (q <= 1.0 && r > 1e-5)
	{
		glm::vec3 gradq = diff * 1.0f / (r * m_h);
		if (q <= 0.5)
		{
			return sigma_d * q * (3 * q - 2) * gradq;
		}
		else
		{
			return -sigma_d * (1 - q) * (1 - q) * gradq;
		}
	}

	return glm::vec3(0, 0, 0);
}

__global__ void update_states_kernel(Particle* particles_device, int num_particles, struct Cell* cells_device
	,struct Box* box_device, float m_h, float m_rho0, float m_gamma, float m_k)
{
	//printf("update states kernel\n");
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= num_particles)
		return;
	
	float new_density = 0;
	float new_pressure = 0;

	auto& particle = particles_device[id];
	glm::ivec3 cell_idx = particle.cell_idx;
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
			{
				glm::ivec3 neighbor_cell_idx = glm::ivec3(i, j, k) + cell_idx;
				if (neighbor_cell_idx.x < 0 || neighbor_cell_idx.y < 0 || neighbor_cell_idx.z < 0 ||
					neighbor_cell_idx.x >= box_device->xres || neighbor_cell_idx.y >= box_device->yres ||
					neighbor_cell_idx.z >= box_device->zres)
					continue;

				int idx = neighbor_cell_idx.x * box_device->yres * box_device->zres
					+ neighbor_cell_idx.y * box_device->zres + neighbor_cell_idx.z;

				for (int cnt = 0; cnt < (cells_device + idx)->count; cnt++)
				{
					int p_idx = (cells_device + idx)->list[cnt];
					if (p_idx < 0 || p_idx >= num_particles)
						continue;
					/*if (id == 0)
						printf("%d\n", p_idx);*/
					new_density += particles_device[p_idx].mass * W(particles_device[p_idx].pos - particle.pos, m_h);
				}
			}

	new_pressure = max(0.f, m_k * m_rho0 / m_gamma * (pow(new_density / m_rho0, m_gamma) - 1.0f));
	particles_device[id].density = new_density;
	particles_device[id].pressure = new_pressure;
	/*if (id == 0)
		printf("%f %f\n", new_pressure, new_density);*/
}

__global__ void compute_force_kernel(Particle* particles_device, int num_particles, struct Cell* cells_device
	, struct Box* box_device, float m_h, float m_mu, int m_fext)
{
	/*printf("compute force kernel\n");
	printf("num of particle %d\n", num_particles);*/
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("particle %d\n", id);
	if (id >= num_particles)
		return;

	auto& particle = particles_device[id];
	glm::vec3 Fp, Fv, Fg, Fext;
	Fp = Fv = Fext = glm::vec3(0.f, 0.f, 0.f);
	Fg = glm::vec3(0.f, 0.f, -9.8 * particle.mass);
	float k_bound = 10000;
	
	glm::ivec3 cell_idx = particle.cell_idx;
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
			{
				glm::ivec3 neighbor_cell_idx = glm::ivec3(i, j, k) + cell_idx;
				if (neighbor_cell_idx.x < 0 || neighbor_cell_idx.y < 0 || neighbor_cell_idx.z < 0 ||
					neighbor_cell_idx.x >= box_device->xres || neighbor_cell_idx.y >= box_device->yres ||
					neighbor_cell_idx.z >= box_device->zres)
					continue;

				int idx = neighbor_cell_idx.x * box_device->yres * box_device->zres
					+ neighbor_cell_idx.y * box_device->zres + neighbor_cell_idx.z;

				for (int cnt = 0; cnt < (cells_device + idx)->count; cnt++)
				{
					int p_idx = (cells_device + idx)->list[cnt];
					auto& neighbor = particles_device[p_idx];
					glm::vec3 grad = gradW(neighbor.pos - particle.pos, m_h);
					Fp +=  grad *
						neighbor.mass / neighbor.density * ((neighbor.pressure + particle.pressure) / 2.f/* + 0.01f * m_h * m_h*/);

					glm::vec3 xij = particle.pos - neighbor.pos;
					glm::vec3 vij = particle.vel - neighbor.vel;
					float tmp = glm::dot(xij, vij);
					float Fvj = 0;
					if (tmp < 0)
						Fvj = neighbor.mass / neighbor.density * glm::dot(xij, vij) / (glm::length2(xij) + 0.01f * m_h * m_h);
					Fv += Fvj * grad;
				}
			}

	Fp = Fp * (particle.mass / particle.density);
	Fv = Fv * (-particle.mass * m_mu * 2 * (3 + 2));

	if (m_fext)
		if (particle.pos.x < 0.65f && particle.pos.x > 0.35f && particle.pos.y < 0.65f && particle.pos.y > 0.35f && particle.pos.z < 0.8f)
			Fext = 3.0f * 9.8f * particle.mass * glm::normalize((glm::vec3(0.5, 0.5, 2.0) - particle.pos));
	
	//boundary
	if (particle.pos.x < box_device->xmin) 
		Fext.x += k_bound * (box_device->xmin - particle.pos.x);
	else if (particle.pos.x > box_device->xmax) 
		Fext.x -= k_bound * (particle.pos.x - box_device->xmax);
	if (particle.pos.y < box_device->ymin)
		Fext.y += k_bound * (box_device->ymin - particle.pos.y);
	else if (particle.pos.y > box_device->ymax)
		Fext.y -= k_bound * (particle.pos.y - box_device->ymax);
	if (particle.pos.z < box_device->zmin)
		Fext.z += k_bound * (box_device->zmin - particle.pos.z);
	else if (particle.pos.z > box_device->zmax)
		Fext.z -= k_bound * (particle.pos.z - box_device->zmax);

	particles_device[id].force = Fp + Fv + Fg + Fext;
	/*if (id == 0)
	{
		printf("%f\n", particles_device[id].mass);
		printf("%f, %f, %f\n", particles_device[id].force.x, particles_device[id].force.y, particles_device[id].force.z);
	}*/
}

__global__ void advection_kernel(Particle* particles_device, int num_particles, float m_dt)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < num_particles)
	{
		particles_device[id].vel += m_dt * particles_device[id].force / particles_device[id].mass;
		particles_device[id].pos += m_dt * particles_device[id].vel;
		/*if (id == 0)
		{
			printf("%f\n", m_dt);
			printf("%f, %f, %f\n", particles_device[id].force.x, particles_device[id].force.y, particles_device[id].force.z);
			printf("%f, %f, %f\n", particles_device[id].vel.x, particles_device[id].vel.y, particles_device[id].vel.z);
		}*/
	}
}