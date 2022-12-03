from enum import Enum
from scipy import ndimage
from PIL import Image
import numpy as np

class Transform(Enum):
    ROTATION = 1
    TRANSLATION_X = 2
    TRANSLATION_Y = 3
    SCALE = 4

EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                            output=jh)

    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
              - np.sum(s2 * np.log(s2)))

    return mi

def transform_image_2d(data, params, param_vals):
    if(Transform.ROTATION in params):
        data = data.rotate(param_vals[params.index(Transform.ROTATION)])
    if(Transform.TRANSLATION_X in params):
        data = data.transform(data.size,
                    Image.Transform.AFFINE, 
                    (1, 0, param_vals[params.index(Transform.TRANSLATION_X)], 0, 1, 0))
    if(Transform.TRANSLATION_Y in params):
        data = data.transform(data.size, 
                    Image.Transform.AFFINE,
                    (1, 0, 0, 0, 1, param_vals[params.index(Transform.TRANSLATION_Y)]))
    return data

def loss(data1, data2, params, param_vals):
    data2 = transform_image_2d(data2, params, param_vals)
    mi = mutual_information_2d(np.asarray(data1).ravel(), np.asarray(data2).ravel())
    return -mi
    
def pso(data1, data2, params, iter=100, random_particles=True, patience=20):
    c1 = 0.1
    c2 = 0.1
    w = 0.8

    width2, height2 = data2.size

    # [[param1, param2, param3], [param1, param2, param3]]
    particles = []
    if(random_particles):
        import random
        
        n_particles = 36
        
        for _ in range(n_particles):
            par_val = [-1]*len(params)

            if(Transform.ROTATION in params):
                par_val[params.index(Transform.ROTATION)] = round(random.uniform(-360, 360), 3)
            if(Transform.TRANSLATION_X in params):
                par_val[params.index(Transform.TRANSLATION_X)] = round(
                    random.uniform(-width2/2, width2/2), 3)
            if(Transform.TRANSLATION_Y in params):
                par_val[params.index(Transform.TRANSLATION_Y)] = round(
                    random.uniform(-height2/2, height2/2), 3)
            particles.append(par_val)
    
    velocities = []
    for _ in range(n_particles):
        vel_val = [-1]*len(params)

        if(Transform.ROTATION in params):
            vel_val[params.index(Transform.ROTATION)] = random.random()*random.random()
        if(Transform.TRANSLATION_X in params):
            vel_val[params.index(Transform.TRANSLATION_X)] = random.random()*random.random()
        if(Transform.TRANSLATION_Y in params):
            vel_val[params.index(Transform.TRANSLATION_Y)] = random.random()*random.random()
        velocities.append(vel_val)

    particles = np.array(particles)
    velocities = np.array(velocities)

    pbest = np.array(particles)
    pbest_obj = np.array([loss(data1, data2, params, x) for x in particles])
    gbest_obj = min(pbest_obj)
    gbest = pbest[np.where(pbest_obj==gbest_obj)[0][0]]

    count_patience = 0
    gbest_obj_before = gbest_obj
    for iteration in range(iter):
        r1 = random.random();
        r2 = random.random();

        # print(np.array([[5, 5, 5], [5, 5, 5]]) - np.array([[1, 1, 1], [3, 3, 3]]))
        # print(np.array([5, 5, 5]) - np.array([[2, 2, 2], [2, 2, 2], [1, 1, 1]]))
        velocities = w * velocities + c1*r1*(pbest - particles) + c2*r2*(gbest-particles)
        particles = particles + velocities

        pbest_obj_temp = np.array([loss(data1, data2, params, x) for x in particles])

        for id_par in range(0, len(particles)):
            if(pbest_obj_temp[id_par]<pbest_obj[id_par]):
                pbest[id_par] = particles[id_par]
                pbest_obj[id_par] = pbest_obj_temp[id_par]

        gbest_obj = min(pbest_obj)
        gbest = pbest[np.where(pbest_obj == gbest_obj)[0][0]]

        print("Iter:", iteration+1, "best param", [x.name for x in params], ":", gbest)
        print("gbest val", gbest_obj, "count patience", count_patience)

        if(gbest_obj != gbest_obj_before):
            count_patience = 0
            gbest_obj_before = gbest_obj
        
        if(count_patience >= patience):
            break
        count_patience += 1
    
    return gbest
        

def findTransformation(data1, data2, params):
    pso(data1, data2, params)
#     print(len(np.asarray(img1).ravel()))
# print(mutual_information_2d(np.asarray(img1).ravel(), np.asarray(img2).ravel()))
#     