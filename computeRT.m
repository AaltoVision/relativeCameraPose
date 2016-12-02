function [R T] = computeRT(cam_file)
    
f = fopen(cam_file, 'r') ;
if f ~= -1
  script=char(fread(f, inf, 'uchar')) ;
  eval(script) ;
  fclose(f);
end

cam_dir = cam_dir;
cam_pos = cam_pos;
cam_up  = cam_up;

z = cam_dir / norm(cam_dir);

x = cross(cam_up,z);
x = x / norm(x);

y = cross(z,x);

R = [x y z];

%T = cam_pos / norm(cam_pos);
T = cam_pos ;

end
