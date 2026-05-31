LoadMass = 0.7;
g = 9.81;

TotalMass = LoadMass + 0.01 * 4;


w_d = [0;0; TotalMass*g;0;0;0];

Fz = LoadMass*g/4