
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <random>
#include <cfloat>


  
//Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "hipsolver.h"
#include "hipblas-export.h"
   


#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>


#include <gmsh.h>


#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
 
struct AABB {
    float3 lower, upper; //not used
};

struct Object {
    AABB aabb;
    uint32_t index;
};

struct Vec3 {
    float x, y, z;

    __host__ __device__
    Vec3 operator-(const Vec3& v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    __host__ __device__
    Vec3 operator+(const Vec3& v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    __host__ __device__
    Vec3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    __host__ __device__
    Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }

    __host__ __device__
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__
    Vec3 cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }
    __host__ __device__
    Vec3 () : x(0), y(0), z(0) {} 

    __host__ __device__
    Vec3 (float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}  

    __host__ __device__
    Vec3 init(float x, float y, float z) {
        Vec3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }

    __host__ __device__
    Vec3 float3ToVec3(const float3& f) {
        return Vec3(f.x, f.y, f.z);
    }

/*
    __host__ __device__
    bool operator==(const Vec3& v) const {
        return (x == v.x) && (y == v.y) && (z == v.z);
    }

    __host__ __device__
    Vec3& operator=(const Vec3& v) {
        if (this != &v) {  
            x = v.x;
            y = v.y;
            z = v.z;
        }
        return *this;
    }
*/
};


struct F3Triangle {
    float3 v0, v1, v2;
};


struct Triangle {
    Vec3 v0, v1, v2;

    __host__ __device__
    bool intersect(const Vec3& rayOrigin, const Vec3& rayDir, float& t) const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = rayDir.cross(edge2);
        float a = edge1.dot(h);

        if (a > -1e-8 && a < 1e-8) return false; // Ray is parallel to triangle

        float f = 1.0f / a;
        Vec3 s = rayOrigin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;

        Vec3 q = s.cross(edge1);
        float v = f * rayDir.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;

        t = f * edge2.dot(q);
        return t > 1e-8; // Intersection occurs
    }
};


struct Box {
    Vec3 min;
    Vec3 max;

    __host__ __device__
    bool intersect(const Vec3& rayOrigin, const Vec3& rayDir, float& t) const {
        float tMin = (min.x - rayOrigin.x) / rayDir.x;
        float tMax = (max.x - rayOrigin.x) / rayDir.x;

        if (tMin > tMax) SWAP(float,tMin, tMax);

        float tyMin = (min.y - rayOrigin.y) / rayDir.y;
        float tyMax = (max.y - rayOrigin.y) / rayDir.y;

        if (tyMin > tyMax) SWAP(float,tyMin, tyMax);

        if ((tMin > tyMax) || (tyMin > tMax))
            return false;

        if (tyMin > tMin)
            tMin = tyMin;

        if (tyMax < tMax)
            tMax = tyMax;

        float tzMin = (min.z - rayOrigin.z) / rayDir.z;
        float tzMax = (max.z - rayOrigin.z) / rayDir.z;

        if (tzMin > tzMax) SWAP(float,tzMin, tzMax);

        if ((tMin > tzMax) || (tzMin > tMax))
            return false;

        t = tMin;
        return true;
    }
};


struct BoxCentroid {
    float3 centroid;
    int index;
};

struct TriangleCentroid {
    float3 centroid;
    int index;
};

struct Rectangle {
    Vec3 v0, v1, v2, v3;
};

struct F3Ray {
    float3 origin;
    float3 direction;
};

struct Ray {
    Vec3 origin, direction;
};


struct BVHNode {
    float3 min, max; // Min and max coordinates of the bounding box
    int leftChild, rightChild; // Indices of child nodes (-1 for leaves)
    int triangleIndex;
    int splitAxis; //add
    int boxIndex; //add
   
};




//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------
//
// Load .OBJ

// Function to load a triangle mesh from a simple OBJ file
bool loadOBJTriangle(const std::string& filename, std::vector<F3Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::vector<float3> vertices;
    std::string line;
    bool isView=false;
    while (std::getline(file, line)) {
        if (line[0] == 'v') {
            float x, y, z;
            sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
            vertices.push_back(make_float3(x, y, z));
            if (isView) std::cout <<"v=<" << x <<","<<y<<","<<z<< ">\n";
        } else if (line[0] == 'f') {
            unsigned int i1, i2, i3;
            sscanf(line.c_str(), "f %u %u %u", &i1, &i2, &i3);
            if (isView) std::cout <<"f=<" << i1 <<","<<i2<<","<<i3<< ">\n";
            triangles.push_back({vertices[i1-1], vertices[i2-1], vertices[i3-1]});
        }
    }
    return true;
}


// Load OBJ Triangle
std::vector<Triangle> loadOBJVec(const std::string& filename)
{
    std::vector<Triangle> triangles;
    std::ifstream file(filename);
    std::string line;
    std::vector<Vec3> vertices;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;


        if (prefix.find("vt") != std::string::npos) { /*...*/ }
        else if (prefix.find("vn") != std::string::npos) { /*...*/ }
        else if (prefix.find("s") != std::string::npos) { /*...*/ }

        else if (prefix == "v") {
            Vec3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        } else if (prefix == "f") {
            int idx0, idx1, idx2;
            iss >> idx0 >> idx1 >> idx2;
            triangles.push_back({vertices[idx0 - 1], vertices[idx1 - 1], vertices[idx2 - 1]});
        }
    }
    return triangles;
}

// Load OBJ Boxes
std::vector<Box> loadBoxes(const std::string& filename)
{
    std::vector<Box> boxes;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Box box;
        if (iss >> box.min.x >> box.min.y >> box.min.z >> box.max.x >> box.max.y >> box.max.z)
        {
            boxes.push_back(box);
        }
    }
    return boxes;
}

//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================

float3 toFloat3(const Vec3& v) { return {v.x, v.y, v.z}; }

std::vector<F3Triangle> boxToTriangles(const Box& box) {
    std::vector<F3Triangle> triangles;
    triangles.reserve(12);

    Vec3 vertices[8] = {
        {box.min.x, box.min.y, box.min.z},
        {box.max.x, box.min.y, box.min.z},
        {box.max.x, box.max.y, box.min.z},
        {box.min.x, box.max.y, box.min.z},
        {box.min.x, box.min.y, box.max.z},
        {box.max.x, box.min.y, box.max.z},
        {box.max.x, box.max.y, box.max.z},
        {box.min.x, box.max.y, box.max.z}
    };

    // Definition of the 12 triangles (2 per face)
    int indices[12][3] = {
        {0, 1, 2}, {0, 2, 3}, // Front face
        {1, 5, 6}, {1, 6, 2}, // Right face
        {5, 4, 7}, {5, 7, 6}, // Back face
        {4, 0, 3}, {4, 3, 7}, // Left face
        {3, 2, 6}, {3, 6, 7}, // Top face
        {4, 5, 1}, {4, 1, 0}  // Bottom face
    };

    for (int i = 0; i < 12; ++i) {
        F3Triangle tri;
        tri.v0 = toFloat3(vertices[indices[i][0]]);
        tri.v1 = toFloat3(vertices[indices[i][1]]);
        tri.v2 = toFloat3(vertices[indices[i][2]]);
        triangles.push_back(tri);
    }
    return triangles;
}


//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================


//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------
//
// Random .OBJ

float randomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen);
}

void buildRandomFileObj(const std::string& filename,int numTriangles) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    file << "# Random" << numTriangles << "Triangles.obj\n";
    file << "# " << numTriangles * 3 << " vertices, " << numTriangles << " triangles\n\n";

    file << "# Vertices\n";
    float3 position;
 

    for (int i = 0; i < numTriangles * 3; ++i) {
        if (i%3==0) { 
            position.x=randomFloat()*100.0;
            position.y=randomFloat()*100.0;
            position.z=randomFloat()*100.0;
        }

        float x = randomFloat()+position.x;
        float y = randomFloat()+position.y;
        float z = randomFloat()+position.z;
        file << "v " << x << " " << y << " " << z << "\n";
    }

    file << "\n# Faces (triangles)\n";
    for (int i = 0; i < numTriangles; ++i) {
        int v1 = i * 3 + 1;
        int v2 = i * 3 + 2;
        int v3 = i * 3 + 3;
        file << "f " << v1 << " " << v2 << " " << v3 << "\n";
    }

    file.close();
}


//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================


//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------
//
// Load .GEO

void loadGeoAndWriteMsh(std::string input, std::string output)
{
    gmsh::initialize();
    gmsh::open(input);

    gmsh::model::mesh::generate(3);  // 3 for maillage 3D

    // Get the mesh nodes
    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords, nodeParams;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

    // Get the mesh elements
    std::vector<int> elemTypes;
    std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
    gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

    // View Informations
    std::cout << "Number of Nodes: " << nodeTags.size() << std::endl;
    std::cout << "Number of elements: " << elemTypes.size() << std::endl;

    // Q ?
    gmsh::write(output);
    std::cout << "Mesh successfully generated and saved to " << output << std::endl;

    gmsh::finalize();
}



struct Mesh {
    std::vector<std::vector<double>> nodes;
    std::vector<std::vector<std::size_t>> elements;
};

std::vector<Mesh> loadGeoFile(const std::string& filename,int dimMesh) {
    gmsh::initialize();
    gmsh::open(filename);
    gmsh::model::mesh::generate(dimMesh);  // 3D mesh

    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords, parametricCoords;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoords);

    std::vector<int> elementTypes;
    std::vector<std::vector<std::size_t>> elementTags, elementNodeTags;
    gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags);


    std::cout << "elementTypes size : " << elementTypes.size() << "\n";
    std::cout << "nodeTags size : " << nodeTags.size() << "\n";

    // Create the Mesh vector
    std::vector<Mesh> meshes;
    for (size_t i = 0; i < elementTypes.size(); ++i) {
        Mesh mesh;
        // Add nodes
        for (size_t j = 0; j < nodeTags.size(); ++j) {
                std::vector<double> node = {nodeCoords[j*3], nodeCoords[j*3+1], nodeCoords[j*3+2]};
                mesh.nodes.push_back(node);
        }

        // Add elements
        mesh.elements.push_back(elementNodeTags[i]);
        meshes.push_back(mesh);
    }

    std::string output= "cubic.msh";
    gmsh::write(output);
    std::cout << "Mesh successfully generated and saved to " << output << std::endl;
    gmsh::finalize();
    return meshes;
}


struct GNode {
    int tag;
    double x, y, z;
};

struct GElement {
    int tag;
    std::vector<int> nodeIndices;
};

struct GMesh {
    std::vector<GNode> nodes;
    std::vector<GElement> elements;
};

std::vector<GMesh> loadMshFile(const std::string& filename,int dimMesh) {
    std::vector<GMesh> meshes;
    gmsh::initialize();
    gmsh::open(filename);

    gmsh::model::mesh::generate(dimMesh);

    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords, nodeParams;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

    std::vector<int> elementTypes;
    std::vector<std::vector<std::size_t>> elementTags, elementNodeTags;
    gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags);

    std::cout << "elementTypes size : " << elementTypes.size() << "\n";
    std::cout << "nodeTags size : " << nodeTags.size() << "\n";

    for (size_t i = 0; i < elementTypes.size(); ++i) {
        GMesh mesh;

        for (size_t j = 0; j < nodeTags.size(); ++j) {
            GNode node;
            node.tag = nodeTags[j];
            node.x = nodeCoords[3*j];
            node.y = nodeCoords[3*j+1];
            node.z = nodeCoords[3*j+2];
            mesh.nodes.push_back(node);
            std::cout <<"tag "<< node.tag <<" ";
            std::cout <<"element ["<<i<<"]"<<" node ["<<j<<"] <"<< node.x <<","<< node.y <<","<< node.z << ">\n";
        }

        for (size_t j = 0; j < elementTags[i].size(); ++j) {
            GElement element;
            element.tag = elementTags[i][j];
            size_t numNodesPerElement = elementNodeTags[i].size() / elementTags[i].size();
            for (size_t k = 0; k < numNodesPerElement; ++k) {
                element.nodeIndices.push_back(elementNodeTags[i][j*numNodesPerElement + k]);
            }
            mesh.elements.push_back(element);
        }

        meshes.push_back(mesh);
    }

    std::string output= "output.msh";
    gmsh::write(output);

    gmsh::finalize();
    return meshes;
}


void loadGeoGmsh(const std::string& filename)
{
    std::vector<Mesh> meshes = loadGeoFile(filename,2);
    for (const auto& mesh : meshes) {
        std::cout << "Number of nodes : " << mesh.nodes.size() << "\n";
        std::cout << "Number of elements : " << mesh.elements.size() <<"\n";
    }
}

void loadMshGmsh(const std::string& filename)
{
    std::vector<GMesh> meshes = loadMshFile(filename,2);
    for (const auto& mesh : meshes) {
        std::cout << "Number of nodes : " << mesh.nodes.size() << "\n";
        std::cout << "Number of elements : " << mesh.elements.size() <<"\n";
    }
}




//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================

__device__ 
float3 calculateTriangleCentroid(const F3Triangle& triangle) {
    return make_float3(
        (triangle.v0.x + triangle.v1.x + triangle.v2.x) / 3.0f,
        (triangle.v0.y + triangle.v1.y + triangle.v2.y) / 3.0f,
        (triangle.v0.z + triangle.v1.z + triangle.v2.z) / 3.0f
    );
}

struct CompareTriangleCentroid {
    int axis;
    CompareTriangleCentroid(int _axis) : axis(_axis) {}
    __device__ bool operator()(const TriangleCentroid& a, const TriangleCentroid& b) const {
        return (axis == 0) ? (a.centroid.x < b.centroid.x) :
            (axis == 1) ? (a.centroid.y < b.centroid.y) :
            (a.centroid.z < b.centroid.z);
    }
};

__device__ float3 calculateBoxCentroid(const Box& box) {
    return make_float3(
        (box.min.x + box.max.x) * 0.5f,
        (box.min.y + box.max.y) * 0.5f,
        (box.min.z + box.max.z) * 0.5f
    );
}

struct CompareBoxCentroidAlpha {
    int axis;
    CompareBoxCentroidAlpha(int _axis) : axis(_axis) {}
    __device__ bool operator()(const BoxCentroid& a, const BoxCentroid& b) const {
        return (axis == 0) ? (a.centroid.x < b.centroid.x) :
            (axis == 1) ? (a.centroid.y < b.centroid.y) :
            (a.centroid.z < b.centroid.z);
    }
};

struct CompareBoxCentroid {
    int axis;
    CompareBoxCentroid(int a) : axis(a) {}
    __device__ bool operator()(const BoxCentroid& a, const BoxCentroid& b) const {
        return (&a.centroid.x)[axis] < (&b.centroid.x)[axis];
    }
};





//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------

//-------       ----------

// With box input


/*
std::vector<Box> generateRandomBoxes(int numBoxes,float m,float M) {
    std::vector<Box> boxes;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(m,M);

    for (int i = 0; i < numBoxes; ++i) {
		Vec3 min,max; 
		min.x=dis(gen);  
		min.y=dis(gen);  
		min.z=dis(gen);
		max.x=min.x + std::abs(dis(gen));
		max.y=min.y + std::abs(dis(gen));
		max.z=min.z + std::abs(dis(gen));
        boxes.push_back({min, max});
    }
    return boxes;
}
*/


void buildBVHWithBox(thrust::device_vector<Box>& boxes, 
                     thrust::device_vector<BVHNode>& nodes,
                     int maxDepth = 20,
                     int minBoxesPerNode = 2,
                     float splitThreshold = 0.5f)
{
    int numBoxes = boxes.size();
    nodes.resize(2 * numBoxes - 1);

    // Calculate centroids and create initial ordering
    thrust::device_vector<BoxCentroid> centroids(numBoxes);
    thrust::transform(thrust::device, boxes.begin(), boxes.end(), centroids.begin(),
        [boxes = thrust::raw_pointer_cast(boxes.data())] __device__ (const Box& box) {
            return BoxCentroid{ calculateBoxCentroid(box), static_cast<int>(&box - boxes) };
        }
    );

    // Lambda to build a subtree
    std::function<void(int, int, int, int)>  buildSubtree = [&](int nodeIndex, int start, int end, int depth) {
        BVHNode* raw_ptr = thrust::raw_pointer_cast(nodes.data());
        BVHNode& node = raw_ptr[nodeIndex];

        thrust::for_each(thrust::device,
            boxes.begin() + start, boxes.begin() + end,
            [&node] __device__ (const Box& box) {
                // Update min
                node.min.x = fminf(node.min.x, box.min.x);
                node.min.y = fminf(node.min.y, box.min.y);
                node.min.z = fminf(node.min.z, box.min.z);
                
                // Update max
                node.max.x = fmaxf(node.max.x, box.max.x);
                node.max.y = fmaxf(node.max.y, box.max.y);
                node.max.z = fmaxf(node.max.z, box.max.z);
            }
        );

        // Leaf node
        if (end - start <= minBoxesPerNode || depth >= maxDepth) {
            //node.triangleIndex = centroids[start].index;
            BoxCentroid* centroid_ptr = thrust::raw_pointer_cast(centroids.data());
            node.boxIndex = centroid_ptr[start].index;
            //node.boxIndex = centroids[start].index;
            node.leftChild = node.rightChild = -1;
            return;
        }

        // Choose split axis (use the axis with the largest extent)
        float3 extent = make_float3(
            node.max.x - node.min.x,
            node.max.y - node.min.y,
            node.max.z - node.min.z
        );
        int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 :
                   (extent.y > extent.z) ? 1 : 2;

        // Sort boxes along the chosen axis
        thrust::sort(thrust::device, centroids.begin() + start, centroids.begin() + end,
            CompareBoxCentroid(axis));

        // Find split point (using splitThreshold)
        int mid = start + static_cast<int>((end - start) * splitThreshold);

        // Recursively build left and right subtrees
        node.leftChild = 2 * nodeIndex + 1;
        node.rightChild = 2 * nodeIndex + 2;
        node.boxIndex = -1;

        buildSubtree(node.leftChild, start, mid, depth + 1);
        buildSubtree(node.rightChild, mid, end, depth + 1);
    };

    // Start building the tree from the root
    buildSubtree(0, 0, numBoxes, 0);
}



__device__ bool intersectRayBox(const F3Ray& ray, const Box& box, float& tmin, float& tmax)
 {
    Vec3 boxMin=Vec3(box.min);
    Vec3 boxMax=Vec3(box.max);

    Vec3 vray_direction; vray_direction.float3ToVec3(ray.direction);
    Vec3 vray_origin;    vray_origin.float3ToVec3(ray.origin);
    Vec3 invDir = Vec3(1.0f / vray_direction.x, 1.0f / vray_direction.y, 1.0f / vray_direction.z);
    Vec3 t0 = (boxMin - vray_origin) * invDir;
    Vec3 t1 = (boxMax - vray_origin) * invDir;
    Vec3 tmin3 = Vec3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    Vec3 tmax3 = Vec3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    tmin = fmaxf(tmin3.x, fmaxf(tmin3.y, tmin3.z));
    tmax = fminf(tmax3.x, fminf(tmax3.y, tmax3.z));
    return tmax >= tmin && tmax > 0;
}


struct RayTracer {
    const BVHNode* bvh;
    const Box* boxes;

    RayTracer(const BVHNode* b, const Box* bx) : bvh(b), boxes(bx) {}

    __device__
    int operator()(const F3Ray& ray) const {
        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0;  
        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            const BVHNode& node = bvh[nodeIdx];
            float tmin, tmax;
            Box node1; 
            Vec3  TMP; 
            TMP.float3ToVec3(node.min);
            node1.min=Vec3(TMP); 
            TMP.float3ToVec3(node.max);
            node1.max=Vec3(TMP); 
            if (intersectRayBox(ray, node1, tmin, tmax)) {
                if (node.boxIndex != -1) {
                    const Box& box = boxes[node.boxIndex];
                    if (intersectRayBox(ray, box, tmin, tmax)) {
                        return node.boxIndex;
                    }
                } else {
                    stack[stackPtr++] = node.leftChild;
                    stack[stackPtr++] = node.rightChild;
                }
            }
        }
        return -1; // No intersection
    }
};


//-------       ----------

//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================







// Intersection function between a ray and a plan
__host__ __device__ std::optional<Vec3> rayPlaneIntersect(const Ray& ray, const Vec3& planePoint, const Vec3& planeNormal) {
    constexpr float epsilon = 1e-6f;

    float denom = planeNormal.dot(ray.direction);

    // Check if the ray is parallel to the plane
    if (fabs(denom) < epsilon) {
        return {}; // No intersection
    }

    Vec3 p0l0 = planePoint - ray.origin;
    float t = p0l0.dot(planeNormal) / denom;

    // Check if the ray is parallel to the plane
    if (t < 0) {
        return {}; // The plan is behind the ray
    }

    // Calculate the intersection point
    return { ray.origin + ray.direction * t };
}

// Intersection function between a ray and a rectangle
__host__ __device__ std::optional<Vec3> rayRectangleIntersect(const Ray& ray, const Rectangle& rect) {
    // We must define the plane of the rectangle
    Vec3 edge1 = rect.v1 - rect.v0;
    Vec3 edge2 = rect.v3 - rect.v0;
    Vec3 normal = edge1.cross(edge2); // Normal of the rectangle

    constexpr float epsilon = 1e-6f;
    float denom = normal.dot(ray.direction);

    // Check if the ray is parallel to the plane
    if (fabs(denom) < epsilon) {
        return {};// No intersection
    }

    // Calculation of the distance t at which the ray intersects the plane
    Vec3 p0l0 = rect.v0 - ray.origin;
    float t = p0l0.dot(normal) / denom;

    // Check if the intersection is in front of the ray
    if (t < 0) {
        return {};// The rectangle is behind the ray
    }

    // The rectangle is behind the ray
    Vec3 intersectionPoint = ray.origin + ray.direction * t;

    // Check if the intersection point is inside the rectangle
    Vec3 c;

    // Check for the first side
    Vec3 edge00 = rect.v1 - rect.v0;
    Vec3 vp0 = intersectionPoint - rect.v0;
    c = edge00.cross(vp0);
    if (normal.dot(c) < 0) return {}; // The point is outside

    // Checking for the second side
    Vec3 edge01 = rect.v2 - rect.v1;
    Vec3 vp1 = intersectionPoint - rect.v1;
    c = edge01.cross(vp1);
    if (normal.dot(c) < 0) return {}; // The point is outside

    // Checking for the third side
    Vec3 edge02 = rect.v3 - rect.v2;
    Vec3 vp2 = intersectionPoint - rect.v2;
    c = edge02.cross(vp2);
    if (normal.dot(c) < 0) return {}; // The point is outside

    // Checking for the fourth side
    Vec3 edge03 = rect.v0 - rect.v3;
    Vec3 vp3 = intersectionPoint - rect.v3;
    c = edge03.cross(vp3);
    if (normal.dot(c) < 0) return {}; // The point is outside

    return intersectionPoint;
}

// Function to calculate the bounding box of a triangle
__host__ __device__
void calculateBoundingBox(const F3Triangle& triangle, float3& min, float3& max)
{
    min = make_float3(fminf(fminf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                      fminf(fminf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                      fminf(fminf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
    max = make_float3(fmaxf(fmaxf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                      fmaxf(fmaxf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                      fmaxf(fmaxf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
}




// Function to build a simple BVH (medium construction method)
void buildBVHWithTriangleVersion1(thrust::device_vector<F3Triangle>& triangles, thrust::device_vector<BVHNode>& nodes)
{
    int numTriangles = triangles.size();
    nodes.resize(2 * numTriangles - 1);

    // Initialize the sheets
    for (int i = 0; i < numTriangles; ++i) {
        BVHNode* raw_ptr = thrust::raw_pointer_cast(nodes.data());
        BVHNode& node = raw_ptr[numTriangles - 1 + i];

        calculateBoundingBox(triangles[i], node.min, node.max);
        node.triangleIndex = i;
        node.leftChild = node.rightChild = -1;
    }

    // Build the internal nodes
    for (int i = numTriangles - 2; i >= 0; --i) {
        BVHNode* raw_ptr = thrust::raw_pointer_cast(nodes.data());
        BVHNode& node = raw_ptr[i];
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;

        node.leftChild = leftChild;
        node.rightChild = rightChild;
        node.triangleIndex = -1;

        BVHNode leftNode = nodes[leftChild];
        BVHNode rightNode = nodes[rightChild];
        node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                               fminf(leftNode.min.y, rightNode.min.y),
                               fminf(leftNode.min.z, rightNode.min.z));


        node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                               fmaxf(leftNode.max.y, rightNode.max.y),
                               fmaxf(leftNode.max.z, rightNode.max.z));
    }
}





//ADDENDUM
void buildBVHWithTriangleVersion2(thrust::device_vector<F3Triangle>& triangles, thrust::device_vector<BVHNode>& nodes)
{
    int numTriangles = triangles.size();
    nodes.resize(2 * numTriangles - 1);

    // Calculate centroids and create initial ordering
    thrust::device_vector<TriangleCentroid> centroids(numTriangles);
    auto centroid_iterator = thrust::make_transform_iterator(
    triangles.begin(),
        [triangles_ptr = thrust::raw_pointer_cast(triangles.data())] __host__ __device__ (const F3Triangle& tri) {
            return TriangleCentroid{ 
                calculateTriangleCentroid(tri), 
                static_cast<int>(&tri - triangles_ptr) // (int)(&tri - triangles_ptr);
            };
            }
    );

    thrust::copy( thrust::device,centroid_iterator, centroid_iterator + triangles.size(), centroids.begin());

    // Lambda to build a subtree
    std::function<void(int, int, int, int)> buildSubtree = [&](int nodeIndex, int start, int end, int depth) {
        BVHNode* raw_ptr = thrust::raw_pointer_cast(nodes.data());
        BVHNode& node = raw_ptr[nodeIndex];

        // Calculate bounding box for this node
        float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = start; i < end; ++i) {
            //const F3Triangle& tri = triangles[centroids[i].index]; 
            const TriangleCentroid* centroid_ptr = thrust::raw_pointer_cast(centroids.data());
            const F3Triangle& tri = triangles[centroid_ptr[i].index];
            calculateBoundingBox(tri, min, max);
        }
        node.min = min;
        node.max = max;

        // Leaf node
        if (end - start <= 1) {
            //node.triangleIndex = centroids[start].index;
            TriangleCentroid* centroid_ptr = thrust::raw_pointer_cast(centroids.data());
            node.triangleIndex = centroid_ptr[start].index;
            node.leftChild = node.rightChild = -1;
            return;
        }

        // Choose split axis (alternate between x, y, z based on depth)
        int axis = depth % 3;

        // Sort triangles along the chosen axis
        thrust::sort(thrust::device, centroids.begin() + start, centroids.begin() + end, CompareTriangleCentroid(axis));

        // Find split point (median)
        int mid = (start + end) / 2;

        // Recursively build left and right subtrees
        node.leftChild = 2 * nodeIndex + 1;
        node.rightChild = 2 * nodeIndex + 2;
        node.triangleIndex = -1;

        buildSubtree(node.leftChild, start, mid, depth + 1);
        buildSubtree(node.rightChild, mid, end, depth + 1);
        };

    // Start building the tree from the root
    buildSubtree(0, 0, numTriangles, 0);
}


//ADDENDUM
// Version 3

__global__ void initializeLeaves(F3Triangle* triangles, BVHNode* nodes, int numTriangles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numTriangles) {
        BVHNode& node = nodes[numTriangles - 1 + i];
        calculateBoundingBox(triangles[i], node.min, node.max);
        node.triangleIndex = i;
        node.leftChild = node.rightChild = -1;
    }
}

__global__ void buildInternalNodes(BVHNode* nodes, int numTriangles)
{
    int i = numTriangles - 2 - (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 0) {
        BVHNode& node = nodes[i];
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;

        node.leftChild = leftChild;
        node.rightChild = rightChild;
        node.triangleIndex = -1;

        BVHNode leftNode = nodes[leftChild];
        BVHNode rightNode = nodes[rightChild];
        node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                               fminf(leftNode.min.y, rightNode.min.y),
                               fminf(leftNode.min.z, rightNode.min.z));

        node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                               fmaxf(leftNode.max.y, rightNode.max.y),
                               fmaxf(leftNode.max.z, rightNode.max.z));
    }
}

void buildBVHWithTriangleVersion3(thrust::device_vector<F3Triangle>& triangles, thrust::device_vector<BVHNode>& nodes)
{
    int numTriangles = triangles.size();
    nodes.resize(2 * numTriangles - 1);
    // Initialize the leaves
    int blockSize = 256;
    int numBlocks = (numTriangles + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(initializeLeaves, dim3(numBlocks), dim3(blockSize), 0, 0, 
        thrust::raw_pointer_cast(triangles.data()),
        thrust::raw_pointer_cast(nodes.data()),
        numTriangles);
    // Build the internal nodes
    numBlocks = ((numTriangles - 1) + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(buildInternalNodes, dim3(numBlocks), dim3(blockSize), 0, 0, thrust::raw_pointer_cast(nodes.data()),numTriangles);
    // Synchronize to ensure all kernels have completed
    hipDeviceSynchronize();
}

//---------------------------------------------------------------------------------------------------------------------


// initialize the leaves
struct InitializeLeavesFunctor
{
    const F3Triangle* triangles;

    InitializeLeavesFunctor(const F3Triangle* tri) : triangles(tri) {}

    __host__ __device__
    BVHNode operator()(int i) const
    {
        BVHNode node;
        calculateBoundingBox(triangles[i], node.min, node.max);
        node.triangleIndex = i;
        node.leftChild = node.rightChild = -1;
        return node;
    }
};

// build the internal nodes
struct BuildInternalNodesFunctor
{
    const BVHNode* nodes;
    int numTriangles;

    BuildInternalNodesFunctor(const BVHNode* n, int num) : nodes(n), numTriangles(num) {}

    __host__ __device__
    BVHNode operator()(int i) const
    {
        i = numTriangles - 2 - i;
        BVHNode node;
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;

        node.leftChild = leftChild;
        node.rightChild = rightChild;
        node.triangleIndex = -1;

        const BVHNode& leftNode = nodes[leftChild];
        const BVHNode& rightNode = nodes[rightChild];
        node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                               fminf(leftNode.min.y, rightNode.min.y),
                               fminf(leftNode.min.z, rightNode.min.z));

        node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                               fmaxf(leftNode.max.y, rightNode.max.y),
                               fmaxf(leftNode.max.z, rightNode.max.z));
        return node;
    }
};

void buildBVHWithTriangleVersion4(thrust::device_vector<F3Triangle>& triangles, thrust::device_vector<BVHNode>& nodes)
{
    int numTriangles = triangles.size();
    nodes.resize(2 * numTriangles - 1);

    // Initialize the leaves
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numTriangles),
        nodes.begin() + (numTriangles - 1),
        InitializeLeavesFunctor(thrust::raw_pointer_cast(triangles.data()))
    );

    // Build the internal nodes
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numTriangles - 1),
        nodes.begin(),
        BuildInternalNodesFunctor(thrust::raw_pointer_cast(nodes.data()), numTriangles)
    );
}

//---------------------------------------------------------------------------------------------------------------------

// Function to make a dot
__host__ __device__
float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Function to make a cross
__host__ __device__
float3 cross(const float3& a, const float3& b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Function to return a length
__host__ __device__
float length(const float3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Function to normalize
__host__ __device__
float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 0) {
        return float3(v.x / len, v.y / len, v.z / len);
    }
    return v;
}

// Function to write a float3
__host__ __device__
void print_float3(const float3& v) {
    printf("%f %f %f\n",v.x,v.y,v.z);
}

__device__ bool rayTriangleIntersect(const F3Ray& ray, const F3Triangle& triangle, float& t, float3& intersectionPoint) {
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (a > -1e-6 && a < 1e-6) return false;

    float f = 1.0f / a;
    float3 s = ray.origin - triangle.v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(edge2, q);

     // Calculate the intersection point
    if (t > 1e-6) {
        intersectionPoint = ray.origin + t * ray.direction;
        //printf("%f %f %f\n",intersectionPoint.x,intersectionPoint.y,intersectionPoint.z); OK
    }
    else
    {
        intersectionPoint =make_float3(INFINITY, INFINITY, INFINITY);
    }

    return (t > 1e-6);
}

__global__ void rayTracingKernel(BVHNode* nodes, F3Triangle* triangles, F3Ray* rays, int* hitResults, float* distance,float3* intersectionPoint, int numRays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    F3Ray ray = rays[idx];
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    float closestT = INFINITY;
    int closestTriangle = -1;
    float3 closestIntersectionPoint=make_float3(INFINITY, INFINITY, INFINITY);
    bool isView=false;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        BVHNode& node = nodes[nodeIdx];

        // Ray-box intersection test
        float tmin = (node.min.x - ray.origin.x) / ray.direction.x;
        float tmax = (node.max.x - ray.origin.x) / ray.direction.x;
        if (tmin > tmax) SWAP(float,tmin, tmax);

        float tymin = (node.min.y - ray.origin.y) / ray.direction.y;
        float tymax = (node.max.y - ray.origin.y) / ray.direction.y;
        if (tymin > tymax) SWAP(float,tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax)) continue;

        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;

        float tzmin = (node.min.z - ray.origin.z) / ray.direction.z;
        float tzmax = (node.max.z - ray.origin.z) / ray.direction.z;
        if (tzmin > tzmax) SWAP(float,tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax)) continue;

        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;

        if (tmax < 0) continue;

        if (node.triangleIndex != -1) {
            // Sheet: test the intersection with the triangle
            float t;
            float3 intersectionPointT;
            if (rayTriangleIntersect(ray, triangles[node.triangleIndex], t,intersectionPointT)) {

                if (isView) printf("[%i] <%f %f %f>\n",idx,intersectionPointT.x,intersectionPointT.y,intersectionPointT.z);

                if (t < closestT) {
                    closestT = t;
                    closestTriangle = node.triangleIndex;
                    closestIntersectionPoint=intersectionPointT;
                }
            }
        } else {
            // Internal node: add children to the stack
            stack[stackPtr++] = node.leftChild;
            stack[stackPtr++] = node.rightChild;
        }
    }

    hitResults[idx]        = closestTriangle;
    distance[idx]          = closestT;
    intersectionPoint[idx] = closestIntersectionPoint;
    //if (closestTriangle!=-1) { printf("t=%f\n",closestT); } OK
}

__global__ void rayTraceKernelBox(Box* boxes, Vec3 rayOrigin, Vec3 rayDir, float* results, int numBoxes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBoxes) {
        float t;
        if (boxes[idx].intersect(rayOrigin, rayDir, t)) {
            results[idx] = t; // Store intersection distance
        } else {
            results[idx] = -1.0f; // No intersection
        }
    }
}

__global__ void rayTraceKernelTriangle(Triangle* triangles, Vec3 rayOrigin, Vec3 rayDir, float* results, int numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numTriangles) {
        float t;
        if (triangles[idx].intersect(rayOrigin, rayDir, t)) {
            results[idx] = t; // Store intersection distance
        } else {
            results[idx] = -1.0f; // No intersection
        }
    }
}

__host__ __device__ std::optional<Vec3> rayTriangleIntersect2(const Ray& ray, const Triangle& triangle) {
    constexpr float epsilon = 1e-6f;

    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 ray_cross_e2 = ray.direction.cross(edge2);
    float det = edge1.dot(ray_cross_e2);

    // Check if the ray is parallel to the triangle
    if (det > -epsilon && det < epsilon) return {};  // No intersection

    float inv_det = 1.0f / det;
    Vec3 s = ray.origin - triangle.v0;
    float u = inv_det * s.dot(ray_cross_e2);

    // Checking barycentric coordinates
    if (u < 0 || u > 1) return {}; // No intersection

    Vec3 s_cross_e1 = s.cross(edge1);
    float v = inv_det * ray.direction.dot(s_cross_e1);

    if (v < 0 || u + v > 1) return {}; // No intersection

    float t = inv_det * edge2.dot(s_cross_e1);

    // Intersection with the ray
    if (t > epsilon)
        return { ray.origin + ray.direction * t }; // Returns the intersection point
    else
        return {}; // No intersection
}

__global__ void rayTriangleIntersectionKernel2(const Ray* rays, const Triangle* triangles, Vec3* intersectionPoints, int numRays, int numTriangles) {
    int rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIndex < numRays) {
        Vec3 intersectionPoint;
        for (int triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++) {
            auto intersection = rayTriangleIntersect2(rays[rayIndex], triangles[triangleIndex]);
            if (intersection) {
                intersectionPoints[rayIndex * numTriangles + triangleIndex] = *intersection;
            }
            else {
                intersectionPoints[rayIndex * numTriangles + triangleIndex] = { 0.0f, 0.0f, 0.0f };
            }
        }
    }
}



//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------

// Only to do a test in normal mode i.e. in CPU
std::optional<Vec3> rayTriangleIntersectCPU(const Ray& ray, const Triangle& triangle) {
    constexpr float epsilon = 1e-6;

    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 pvec = ray.direction.cross(edge2);
    float det = edge1.dot(pvec);

    // Check if the ray is parallel to the triangle
    if (det > -epsilon && det < epsilon) {
        return {}; // No intersection
    }

    float inv_det = 1.0 / det;

    Vec3 tvec = ray.origin - triangle.v0;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0 || u > 1) {
        return {}; // No intersection
    }

    Vec3 qvec = tvec.cross(edge1);
    float v = ray.direction.dot(qvec) * inv_det;
    if (v < 0 || u + v > 1) {
        return {}; // No intersection
    }

    float t = edge2.dot(qvec) * inv_det;
    // Intersection with the ray
    if (t > epsilon) {
        return { ray.origin + Vec3{ray.direction.x * t, ray.direction.y * t, ray.direction.z * t} };
    }
    else {
        return {};// No intersection
    }
}


//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================


//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------
// Morton 3D Algorithm

//...

//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================





//===========================================================================================================================================
//-------------------------------------------------------------------------------------------------------------------------------------------

// BEGIN:: BVH and Ray Tracing part
void test001_IntersectionRayWithSurfaceCPU()
{
    Ray ray = { {0.5, 0.5, -0.5}, {0, 0, 1} };
    Rectangle rect = { {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0} };

    auto intersection = rayRectangleIntersect(ray, rect);
    if (intersection) {
        std::cout << "Intersection Point: ("
            << intersection->x << ", "
            << intersection->y << ", "
            << intersection->z << ")\n";
    }
    else {
        std::cout << "No intersection.\n";
    }
}

void test001_IntersectionRayWithTriangleCPU()
{
    Ray ray = { {0, 0, 0}, {1, 1, 0} };
    Triangle triangle = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };

    auto intersection = rayTriangleIntersectCPU(ray, triangle);
    if (intersection) {
        std::cout << "Point Intersection: ("
            << intersection->x << ", "
            << intersection->y << ", "
            << intersection->z << ")\n";
    }
    else {
        std::cout << "No intersection.\n";
    }
}


void test001_IntersectionRayWithPlanCPU()
{
    Ray ray = { {0, 0, 0}, {1, 1, 0} };
    Vec3 planePoint = { 0, 0, 1 };  // Position
    Vec3 planeNormal = { 0, 0, 1 }; // Normal

    auto intersection = rayPlaneIntersect(ray, planePoint, planeNormal);
    if (intersection) {
        std::cout << "Point Intersection: ("
            << intersection->x << ", "
            << intersection->y << ", "
            << intersection->z << ")\n";
    }
    else {
        std::cout << "No intersection.\n";
    }
}


// Now in GPU AMD HIP with rocThrust

void test002_with_triangle(const std::string& filename)
{

    std::cout<<"\n";
    std::cout<<"[INFO]: BEGIN BVH WITH TRIANGLE\n";
    std::cout<<"[INFO]: LOAD SCENE >>>"<<filename<<"\n";

    // Load the mesh
    std::vector<F3Triangle> hostTriangles;
    loadOBJTriangle(filename, hostTriangles);
    //getchar();
    // Formulation triangle in file v %f %f %f  ... %u %u %u   <float3,float,float> and index
      
    // Transfer triangles to GPU
    thrust::device_vector<F3Triangle> deviceTriangles = hostTriangles;

    // Building the BVH
    thrust::device_vector<BVHNode> deviceNodes;
    buildBVHWithTriangleVersion1(deviceTriangles, deviceNodes);
    //buildBVHWithTriangleVersion2(deviceTriangles, deviceNodes);
    //buildBVHWithTriangleVersion3(deviceTriangles, deviceNodes);
    //buildBVHWithTriangleVersion4(deviceTriangles, deviceNodes);


    std::cout<<"[INFO]: BVH built with " << deviceNodes.size() << " nodes" << std::endl;

    // Generate test several rays
    std::cout<<"[INFO]: GENERATE TEST SEVERAL RAYS\n";
    const int numRays = 1024;

    thrust::host_vector<float3> hostIntersectionPoint(numRays);
    
    thrust::host_vector<F3Ray> hostRays(numRays);
    for (int i = 0; i < numRays; ++i) {
        hostRays[i].origin = make_float3(0, 0, 10);
        hostRays[i].direction = make_float3(
            (float)rand() / RAND_MAX * 2 - 1,
            (float)rand() / RAND_MAX * 2 - 1,
            -1
        );
        hostRays[i].direction = normalize(hostRays[i].direction);
        hostIntersectionPoint[i]=make_float3(INFINITY, INFINITY, INFINITY);
    }
    thrust::device_vector<F3Ray>  deviceRays              = hostRays;
    thrust::device_vector<float3> deviceIntersectionPoint = hostIntersectionPoint;

    // Allocate memory for the results
    thrust::device_vector<int>   deviceHitResults(numRays);
    thrust::device_vector<float> deviceDistanceResults(numRays);

    // Start the ray tracing kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;
    rayTracingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(deviceNodes.data()),
        thrust::raw_pointer_cast(deviceTriangles.data()),
        thrust::raw_pointer_cast(deviceRays.data()),
        thrust::raw_pointer_cast(deviceHitResults.data()),
        thrust::raw_pointer_cast(deviceDistanceResults.data()),
        thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
        numRays
    );
    //TODO add parameter distance

    // Retrieve the results
    thrust::host_vector<int>   hostHitResults = deviceHitResults;
    thrust::host_vector<float> hostDistanceResults = deviceDistanceResults;
    hostIntersectionPoint=deviceIntersectionPoint;

    // Count intersections
    int hitCount = thrust::count_if(hostHitResults.begin(), hostHitResults.end(),thrust::placeholders::_1 != -1);
    std::cout<<"[INFO]: Number of rays that intersected the mesh : " << hitCount << " / " << numRays << std::endl;
    for (int i=0;i<hostHitResults.size();++i)
    {
        if (hostHitResults[i]!=-1)
        {
            std::cout<<"      Intersection found with Num Ray ["<<i<<"] ori= <"<<hostRays[i].origin.x<<","<<hostRays[i].origin.y<<","<<hostRays[i].origin.z<<"> ";
            std::cout<<" dir= <"<<hostRays[i].direction.x<<","<<hostRays[i].direction.y<<","<<hostRays[i].direction.z<<"> ";
            std::cout<<" Hit= "<<hostHitResults[i]<<" min dist="<<hostDistanceResults[i];
            std::cout<<" IntersectionPoint= <"<<hostIntersectionPoint[i].x<<","<<hostIntersectionPoint[i].y<<","<<hostIntersectionPoint[i].z<<"> "<<"\n";
        }
    }
    std::cout<<"\n";
    std::cout<<"[INFO]: END BVH WITH TRIANGLE\n";
    std::cout<<"\n";
}


void test003_with_Simple_Box(const std::string& filename)
{
    std::cout<<"\n";
    std::cout<<"[INFO]: BEGIN BVH WITH SIMPLE BOX\n";
    // Load the mesh from an OBJ file
    std::vector<Box> boxes = loadBoxes(filename);
    // Formulation box; box.min.x box.min.y box.min.z  box.max.x box.max.y box.max.z used in Feelpp BVH

    // Copy triangles to GPU
    thrust::device_vector<Box> d_boxes(boxes.begin(), boxes.end());
    thrust::device_vector<float> d_results(boxes.size());

    // Define Ray
    Vec3 rayOrigin = {-9.0f, 0.0f, 0.0f};
    Vec3 rayDir = {1.0f, 0.0f, 0.0f};

    // Start the ray tracing kernel
    int numBoxes = boxes.size();
    int blockSize = 256;
    int numBlocks = (numBoxes + blockSize - 1) / blockSize;
    rayTraceKernelBox<<<numBlocks, blockSize>>>(d_boxes.data().get(), rayOrigin, rayDir, d_results.data().get(), numBoxes);

    // Copy results from GPU to CPU
    thrust::host_vector<float> results = d_results;

    // Show results
    for (int i = 0; i < numBoxes; ++i) {
        if (results[i] >= 0) {
            std::cout << "[INFO]: Intersection found with box " << i << " à t = " << results[i] << std::endl;
        }
    }

    std::cout<<"\n";
    std::cout<<"[INFO]: END BVH WITH SIMPLE BOX";
    std::cout<<"\n";
}



void test004_with_Box(const std::string& filename)
{
    std::cout<<"\n";
    std::cout<<"[INFO]: BEGIN BVH WITH BOX\n";
    // Load the mesh from an OBJ file
    std::vector<Box> boxes = loadBoxes(filename);
    thrust::device_vector<Box> deviceBoxes = boxes;
    thrust::device_vector<BVHNode> deviceNodes;
    buildBVHWithBox(deviceBoxes, deviceNodes);
    std::cout << "Number of BVH nodes created: " << deviceNodes.size() << std::endl;

	const int numRays = 10;
    thrust::host_vector<F3Ray> hostRays(numRays);
    for (int i = 0; i < numRays; ++i) {
        hostRays[i].origin = make_float3(0, 0, 10);
        hostRays[i].direction = make_float3(
            (float)rand() / RAND_MAX * 2 - 1,
            (float)rand() / RAND_MAX * 2 - 1,
            -1
        );
        hostRays[i].direction = normalize(hostRays[i].direction);
    }
    thrust::device_vector<F3Ray>  deviceRays = hostRays;

    RayTracer tracer(thrust::raw_pointer_cast(deviceNodes.data()),
                     thrust::raw_pointer_cast(deviceBoxes.data()));
     
	thrust::device_vector<int> results(numRays);
    thrust::transform(deviceRays.begin(), deviceRays.end(), results.begin(), tracer);

	// Results
    thrust::host_vector<int> hostResults = results;
    for (int i = 0; i < numRays; ++i) {
        if (hostResults[i] != -1) {
            std::cout << "Ray " << i << " touched the index box : " << hostResults[i] << std::endl;
        } else {
            std::cout << "Ray " << i << " no hit !" << std::endl;
        }
    }

    std::cout<<"\n";
    std::cout<<"[INFO]: END BVH WITH BOX";
    std::cout<<"\n";
}


// END:: BVH and Ray Tracing part

//-------------------------------------------------------------------------------------------------------------------------------------------
//===========================================================================================================================================

  


  
 

int main(){
    int count, device;
    hipGetDeviceCount(&count);
    hipGetDevice(&device);
    printf("TRIVIAL TEST %d %d \n", device, count);
    std::cout << "[INFO]: WELL DONE :-) GPU AMD!" << "\n";

    std::string filename= "cubic.obj";
    
    loadGeoGmsh(filename);
    std::cout << "[INFO]: WELL DONE :-) LOAD .GEO!" << "\n";

    std::cout << "------------------------------------------------------------------------------------------------" << "\n";

    filename= "cubic2.msh";
    loadMshGmsh(filename);
    std::cout << "[INFO]: WELL DONE :-) LOAD .MSH!" << "\n";
    std::cout << "------------------------------------------------------------------------------------------------" << "\n";



    test001_IntersectionRayWithSurfaceCPU();
    std::cout << "[INFO]: WELL DONE :-) IntersectionRayWithSurface!" << "\n";
    
    std::cout << "------------------------------------------------------------------------------------------------" << "\n";
    
    test001_IntersectionRayWithTriangleCPU();
    std::cout << "[INFO]: WELL DONE :-) IntersectionRayWithTriangle!" << "\n";

    std::cout << "------------------------------------------------------------------------------------------------" << "\n";
    filename = "Triangle.obj";
    test002_with_triangle(filename);
    std::cout << "[INFO]: WELL DONE :-)!" << "\n";


    std::cout << "------------------------------------------------------------------------------------------------" << "\n";
    filename = "Triangle2Cube.obj";
    test002_with_triangle(filename);
    std::cout << "[INFO]: WELL DONE :-)" << "\n";


    std::cout << "------------------------------------------------------------------------------------------------" << "\n";
    filename = "Box.obj";
    test003_with_Simple_Box(filename);
    std::cout << "[INFO]: WELL DONE :-) Box!"<<"\n";
    std::cout << "------------------------------------------------------------------------------------------------" << "\n";

    filename = "Box.obj";
    test004_with_Box(filename);
    std::cout << "[INFO]: WELL DONE :-) Part 4!"<<"\n";

    std::cout << "------------------------------------------------------------------------------------------------" << "\n";

    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";

    //buildRandomFileObj("random.obj",1000);

    return 0;
}



