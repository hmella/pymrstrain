// Dimensions
lc = 0.025;   // characteristic length
L = 1.0;   // length
R = 1.0;    // radius

// Points
Point(1) = {0, 0, -0.5*L, lc};
Point(2) = {-R, 0, -0.5*L, lc};
Point(3) = {0, -R, -0.5*L, lc};
Point(5) = {R, 0, -0.5*L, lc};
Point(6) = {0, R, -0.5*L, lc};

// Circle arcs
Circle(1) = {6, 1, 2};
Circle(2) = {2, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 6};

// Add surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Extrude surface
out[] = Extrude {0, 0, L}{ 
  Surface{1}; 
  Layers{Round(L/lc)};
};

// Whole domain
Physical Volume(1) = {out[1]};

// Labels
Physical Surface(1) = {1};      // inlet
Physical Surface(2) = {out[0]}; // outlet
Physical Surface(3) = {out[2],out[3],out[4],out[5]};  // walls

Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;