clear all
clc

% type of detector
orb = 1;
surf = 0;

% path to the images of lr3 dataset
source_data_path = './data/living_room_traj3_loop';
% and txt files with cameras ground truth
ds = imageDatastore(source_data_path, 'FileExtensions', '.png');
ds_camera_params = tabularTextDatastore(source_data_path, 'FileExtensions', '.txt');
number_of_images = length(ds_camera_params.Files);

% load GT adjancy matrix 
adj_mtx_depth_gt = fread(fopen('./data/ajd_mtx_depth.bin', 'r'), [number_of_images number_of_images], 'int');
U = triu(adj_mtx_depth_gt - diag(diag(adj_mtx_depth_gt)));

% array of matched points (inliers) in two images
matched_points_coordinates = cell(number_of_images);

% initialize 2 matrices (arrays) for relative orientation and
% translation errors (in degree). Both of them are diagonal.
orientation_err = -1 .* ones(number_of_images, number_of_images);
orientation_err = orientation_err - diag(diag(orientation_err));
translation_err = -1 .* ones(number_of_images, number_of_images);
translation_err = translation_err - diag(diag(translation_err));

% convert matched camera pairs to index representation
matched_idxs = find(U);
[I,J] = ind2sub([number_of_images, number_of_images], matched_idxs);
IND = [I, J];
[~, s_ids] = sort(I);
IND_sorted = IND(s_ids, :);

% For those cases essential matrix can not be calculated
bad_cases = zeros(size(IND_sorted, 1), 3);

if (orb)
    addpath('/home/mexopencv-master');
    % create detector
    detector = cv.ORB();
    % create matcher
    matcher = cv.DescriptorMatcher('BFMatcher', 'NormType', 'Hamming', 'CrossCheck', true);
end

% id of the first camera
prev_id1 = 0;

% iterate throw the all matched camera pairs
for k = 1:size(IND_sorted, 1)
    curnt_id1 = IND_sorted(k, 1);
    curnt_id2 = IND_sorted(k, 2);
    
    % to preserve the computation time we compute detectors if camera_id1
    % on the current step is not equal camera_id1 on the previous.
    if (prev_id1 ~= curnt_id1)
        K1 = getcamK(ds_camera_params.Files{curnt_id1});
        [R1, T1] = computeRT(ds_camera_params.Files{curnt_id1});
        
        i1_grey = rgb2gray(imread(ds.Files{curnt_id1}));
        %i1_grey = rgb2gray(cv.imread(ds.Files{curnt_id1}));
        
        if (orb)
            [kp1, des1] = detector.detectAndCompute(i1_grey);
            image_points_1 = reshape([kp1.pt], 2, []).';
        elseif (surf)
            cameraParams1 = cameraParameters('IntrinsicMatrix', K1.');
            image_points_1 = detectSURFFeatures(i1_grey);
            % extract descriptors
            des1 = extractFeatures(i1_grey, image_points_1);
        end
        
        prev_id1 = curnt_id1;
    end
    K2 = getcamK(ds_camera_params.Files{curnt_id2});
    [R2, T2] = computeRT(ds_camera_params.Files{curnt_id2});
    i2_grey = rgb2gray(imread(ds.Files{curnt_id2}));
    %i2_grey = rgb2gray(cv.imread(ds.Files{curnt_id2}));
    
    if (orb)
        [kp2, des2] = detector.detectAndCompute(i2_grey);
        image_points_2 = reshape([kp2.pt], 2, []).';
        
        % checking #1
        if (isempty(des1) || isempty(des2))
            err_code = -3;
            bad_cases(k, :) = [IND_sorted(k, :), err_code];
            continue;
        elseif (isempty(kp1) || isempty(kp2))
            err_code = -4;
            bad_cases(k, :) = [IND_sorted(k, :), err_code];
            continue;
        end
        
        matchedPairs = matcher.match(des1, des2);
        
        % ids of matched points in image1
        matched_pnt_ids_img_1 = ([matchedPairs.queryIdx] + 1)';
        % ids of matched points in image2
        matched_pnt_ids_img_2 = ([matchedPairs.trainIdx] + 1)';
        % get sorted matched points' ids in asceding order 
        % (according to distance between points)
        [~, dist_ids] = sort([matchedPairs.distance]);
        % sort matched points for both images
        matched_pnt_ids_img_1 = matched_pnt_ids_img_1(dist_ids);
        matched_pnt_ids_img_2 = matched_pnt_ids_img_2(dist_ids);
        % get sorted match points (coordinates)
        matched_pnt_img_1 = image_points_1(matched_pnt_ids_img_1, :);
        matched_pnt_img_2 = image_points_2(matched_pnt_ids_img_2, :);
        
        matched_points_coordinates{curnt_id1, curnt_id2} = [matched_pnt_img_1 ...
                                                            matched_pnt_img_2];
        % calculate essential matrix                                                
        [E, mask_eM] = cv.findEssentialMat(matched_pnt_img_1, matched_pnt_img_2, ...
                                    'CameraMatrix', K2, 'Method','Ransac');
                                
        inliers_img_1 = matched_pnt_img_1(logical(mask_eM), :);
        inliers_img_2 = matched_pnt_img_2(logical(mask_eM), :);
        
        % checking #2
        if (isempty(inliers_img_1) || isempty(inliers_img_2))
            err_code = 1;
            bad_cases(k, :) = [IND_sorted(k, :), err_code];
            continue;
        elseif ((size(inliers_img_1, 1) < 5) || (size(inliers_img_2, 1) < 5))
            err_code = 2;
            bad_cases(k, :) = [IND_sorted(k, :), err_code];
            continue;
        end
        
        % calculate relative orientation and translation of two cameras
        [delta_R_est, delta_T_est, good, mask_rP] = cv.recoverPose(E, inliers_img_1, inliers_img_2, 'CameraMatrix', K2);
        
    elseif (surf)
        cameraParams2 = cameraParameters('IntrinsicMatrix', K2.');
        image_points_2 = detectSURFFeatures(i2_grey);
        % extract descriptors
        des2 = extractFeatures(i2_grey, image_points_2);
        
        indexPairs = matchFeatures(des1, des2);
        matchedPoints1 = image_points_1(indexPairs(:,1));
        matchedPoints2 = image_points_2(indexPairs(:,2));
        
        % calculate essential matrix
        [E, inliers, status] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, cameraParams1, cameraParams2);
        if (status == 0)
            inliers_img_1 = matchedPoints1(inliers);
            inliers_img_2 = matchedPoints2(inliers);
            [est_rel_orient, est_rel_location] = relativeCameraPose(E, cameraParams1, cameraParams2, inliers_img_1, inliers_img_2);
            % calculate relative orientation and translation of two cameras
            [delta_R_est, delta_T_est] = cameraPoseToExtrinsics(est_rel_orient, est_rel_location);
        else
            bad_cases(k, :) = [IND_sorted(k, :), status];
        end
    end
    
    % relative orientation GT
    delta_R_gt = R1 * R2.';
    % normalize translation GT for both cameras
    t1_norm = T1 / norm(T1);
    t2_norm = T2 / norm(T2);
    
    % relative translation GT
    delta_T_gt_so_far = t1_norm - (delta_R_gt * t2_norm / norm(delta_R_gt * t2_norm));
    % relative translation (L2 normalized)
    delta_T_gt = delta_T_gt_so_far / norm(delta_T_gt_so_far);
    
    % calculate relative orientation error
    delta_R = delta_R_est.' * delta_R_gt;
    orient_angular_err = norm(rotationpars(delta_R));
    trans_angular_err = acos(dot(delta_T_est, delta_T_gt));
        
    orientation_err(curnt_id1, curnt_id2) = rad2deg(orient_angular_err);
    orientation_err(curnt_id2, curnt_id1) = rad2deg(orient_angular_err);
    translation_err(curnt_id1, curnt_id2) = rad2deg(trans_angular_err);
    translation_err(curnt_id2, curnt_id1) = rad2deg(trans_angular_err);
    

end
bad_cases( ~any(bad_cases,2), : ) = [];