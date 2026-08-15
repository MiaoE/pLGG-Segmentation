def sam_segmentation_mask(mask_generator, image):
    masks = mask_generator.generate(image)
    '''output format
    [{  'segmentation': array([[False,  True,  True, ...,  True, False, False],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        ...,
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True]], shape=(240, 240)), 
        'area': 57574, 
        'bbox': [0, 0, 239, 239], 
        'predicted_iou': 1.0361067056655884, 
        'point_coords': [[236.25, 183.75]], 
        'stability_score': 0.9902777671813965, 
        'crop_box': [0, 0, 240, 240]
    }]
    '''
    return masks

def sam_segmentation_with_bbox(predictor, image, bbox):
    '''expects image to be 3 channels (W, H, 3)
    bbox in format [x,y,x,y] (single array)
    
    :param predictor: SAM predictor object
    :param image: the MRI layer image in 3 channels
    :param bbox: numpy array in [x0, y0, x1, y1] format (single array)
    '''
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=bbox[None, :], multimask_output=True)
    best_mask = masks[scores.argmax()]
    # masks, _, _ = predictor.predict(box=bbox[None, :], multimask_output=False)
    '''output format
    [[[False False False ... False False False]
      [False False False ... False False False]
      [False False False ... False False False]
      ...
      [False False False ... False False False]
      [False False False ... False False False]
      [False False False ... False False False]]]'''
    return best_mask

def sam_segmentation_with_point(predictor, image, point, label):
    '''
    :param predictor: SAM predictor object
    :param image: the MRI layer image
    :param point: numpy array in [[x, y]] format (double array)
    :param label: numpy single element array, 1 - foreground point, 0 - background point 
    '''
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
    return masks