function K = getcamK(cam_file)

f = fopen(cam_file, 'r') ;
if f ~= -1
  script=char(fread(f, inf, 'uchar')) ;
  eval(script) ;
  fclose(f) ;
end

focal  =         norm(cam_dir) ;
aspect =         norm(cam_right)   / norm(cam_up) ;
angle  = 2*atan( norm(cam_right)/2 / norm(cam_dir)  ) ;


M      = 480; %cam_height 
N      = 640; %cam.width 

width = N;
height = M;

% pixel size 
psx = 2*focal*tan(0.5*angle)/N ;
psy = 2*focal*tan(0.5*angle)/aspect/M ;

psx   = psx / focal; 
psy   = psy / focal ;
% 
 Sx = psx;
 Sy = psy;

Ox = (width+1)*0.5;
Oy = (height+1)*0.5;

f = focal;

K = [1/psx     0     Ox;
            0    1/psy    Oy;
            0      0     1];
        
K(2,2) = -K(2,2);        

end
