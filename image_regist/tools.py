from enum import Enum
from scipy import ndimage
from PIL import Image
import random
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


def cross_entropy_loss(x, y, sigma=1):
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

    mi = np.sum(s1 * np.log(s2))

    return -mi


def derivative_ce(x, y, sigma=1):
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

    return -(s1.ravel()[0]/s2.ravel()[0])


def transform_image_2d(data, params, param_vals):
    if (Transform.ROTATION in params):
        data = data.rotate(param_vals[params.index(Transform.ROTATION)])
    if (Transform.TRANSLATION_X in params):
        data = data.transform(data.size,
                              Image.Transform.AFFINE,
                              (1, 0, param_vals[params.index(Transform.TRANSLATION_X)], 0, 1, 0))
    if (Transform.TRANSLATION_Y in params):
        data = data.transform(data.size,
                              Image.Transform.AFFINE,
                              (1, 0, 0, 0, 1, param_vals[params.index(Transform.TRANSLATION_Y)]))
    return data


def loss(data1, data2, params, param_vals, faster_sample=None):
    data2 = transform_image_2d(data2, params, param_vals)
    if (faster_sample == None):
        mi = mutual_information_2d(np.asarray(
            data1).ravel(), np.asarray(data2).ravel())
    else:
        d1 = np.take(data1, faster_sample)
        d2 = np.take(data2, faster_sample)
        mi = mutual_information_2d(np.asarray(
            d1).ravel(), np.asarray(d2).ravel())

    return -mi


def pso(data1, data2, params, faster=False, iter=100, random_particles=True, patience=10):
    c1 = 0.1
    c2 = 0.1
    w = 0.8

    width2, height2 = data2.size

    if (faster):
        faster_sample = random.sample(
            range(0, width2*height2), round(width2*height2*0.2))
    else:
        faster_sample = None

    # [[param1, param2, param3], [param1, param2, param3]]
    particles = []
    if (random_particles):

        n_particles = 25
        for _ in range(n_particles):
            par_val = [-1]*len(params)

            if (Transform.ROTATION in params):
                par_val[params.index(Transform.ROTATION)] = round(
                    random.uniform(-360, 360), 3)
            if (Transform.TRANSLATION_X in params):
                par_val[params.index(Transform.TRANSLATION_X)] = round(
                    random.uniform(-width2/2, width2/2), 3)
            if (Transform.TRANSLATION_Y in params):
                par_val[params.index(Transform.TRANSLATION_Y)] = round(
                    random.uniform(-height2/2, height2/2), 3)
            particles.append(par_val)

    velocities = []
    for _ in range(n_particles):
        vel_val = [-1]*len(params)

        if (Transform.ROTATION in params):
            vel_val[params.index(Transform.ROTATION)
                    ] = random.random()*random.random()
        if (Transform.TRANSLATION_X in params):
            vel_val[params.index(Transform.TRANSLATION_X)
                    ] = random.random()*random.random()
        if (Transform.TRANSLATION_Y in params):
            vel_val[params.index(Transform.TRANSLATION_Y)
                    ] = random.random()*random.random()
        velocities.append(vel_val)

    particles = np.array(particles)
    velocities = np.array(velocities)

    pbest = np.array(particles)
    pbest_obj = np.array(
        [loss(data1, data2, params, x, faster_sample) for x in particles])
    gbest_obj = min(pbest_obj)
    gbest = pbest[np.where(pbest_obj == gbest_obj)[0][0]]

    count_patience = 0
    gbest_obj_before = gbest_obj
    for iteration in range(iter):
        r1 = random.random()
        r2 = random.random()

        # print(np.array([[5, 5, 5], [5, 5, 5]]) - np.array([[1, 1, 1], [3, 3, 3]]))
        # print(np.array([5, 5, 5]) - np.array([[2, 2, 2], [2, 2, 2], [1, 1, 1]]))
        velocities = w * velocities + c1*r1 * \
            (pbest - particles) + c2*r2*(gbest-particles)
        particles = particles + velocities

        pbest_obj_temp = np.array(
            [loss(data1, data2, params, x, faster_sample) for x in particles])

        for id_par in range(0, len(particles)):
            if (pbest_obj_temp[id_par] < pbest_obj[id_par]):
                pbest[id_par] = particles[id_par]
                pbest_obj[id_par] = pbest_obj_temp[id_par]

        gbest_obj = min(pbest_obj)
        gbest = pbest[np.where(pbest_obj == gbest_obj)[0][0]]

        print("Iter:", iteration+1, "best param",
              [x.name for x in params], ":", gbest)
        # print("gbest val", gbest_obj, "count patience", count_patience)

        if (round(gbest_obj, 3) != round(gbest_obj_before, 3)):
            count_patience = 0
            gbest_obj_before = gbest_obj

        if (count_patience >= patience):
            break

        # robustness
        if (count_patience > patience - 4):
            for particle in particles:
                flag_rand = random.randint(0, 4)
                if (flag_rand == 0 and Transform.ROTATION in params):
                    particle[params.index(Transform.ROTATION)] -= 180
                if (flag_rand == 1 and Transform.TRANSLATION_X in params):
                    particle[params.index(Transform.TRANSLATION_X)] *= -1
                if (flag_rand == 2 and Transform.TRANSLATION_Y in params):
                    particle[params.index(Transform.TRANSLATION_Y)] *= -1

        count_patience += 1

    return gbest


def derivative_function(data1, data2, params, param_vals):
    par_val_d = np.copy(param_vals)
    if (Transform.ROTATION in params):
        d2_tranformed = transform_image_2d(data2, [Transform.ROTATION],
                                           [param_vals[params.index(Transform.ROTATION)]])

        par_val_d[params.index(Transform.ROTATION)] = 0.01*derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    if (Transform.TRANSLATION_X in params):
        d2_tranformed = transform_image_2d(data2, [Transform.TRANSLATION_X],
                                           [param_vals[params.index(Transform.TRANSLATION_X)]])

        par_val_d[params.index(Transform.TRANSLATION_X)] = 0.001*derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    if (Transform.TRANSLATION_Y in params):
        d2_tranformed = transform_image_2d(data2, [Transform.TRANSLATION_Y],
                                           [param_vals[params.index(Transform.TRANSLATION_Y)]])

        par_val_d[params.index(Transform.TRANSLATION_Y)] = 0.001*derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    return par_val_d


def gd(data1, data2, params):
    rate = 1
    iteration = 200

    par_vals = [-1]*len(params)
    width2, height2 = data2.size

    if (Transform.ROTATION in params):
        par_vals[params.index(Transform.ROTATION)] = round(
            random.uniform(-360, 360), 3)
    if (Transform.TRANSLATION_X in params):
        par_vals[params.index(Transform.TRANSLATION_X)] = round(
            random.uniform(-width2/8, width2/8), 3)
    if (Transform.TRANSLATION_Y in params):
        par_vals[params.index(Transform.TRANSLATION_Y)] = round(
            random.uniform(-height2/8, height2/8), 3)

    best_param = np.copy(par_vals)
    for i in range(0, iteration):
        loss_val = loss(data1, data2, params, best_param)

        param_val_derivative = derivative_function(
            data1, data2, params, best_param)

        best_param[0] = best_param[0] - rate * param_val_derivative[0]
        best_param[1] = best_param[1] - rate * param_val_derivative[1]
        best_param[2] = best_param[2] - rate * param_val_derivative[2]

        print("Iter:", i+1, "best param",
              [x.name for x in params], ":", best_param)
        print("derivative", param_val_derivative)
        print("loss", loss_val)

    return best_param


def gradient_powell(data1, data2, params, param_vals):
    par_val_d = np.copy(param_vals)
    if (Transform.ROTATION in params):
        d2_tranformed = transform_image_2d(data2, [Transform.ROTATION],
                                           [param_vals[params.index(Transform.ROTATION)]])

        par_val_d[params.index(Transform.ROTATION)] = 0.1*derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    if (Transform.TRANSLATION_X in params):
        d2_tranformed = transform_image_2d(data2, [Transform.TRANSLATION_X],
                                           [param_vals[params.index(Transform.TRANSLATION_X)]])

        par_val_d[params.index(Transform.TRANSLATION_X)] = 0.1* derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    if (Transform.TRANSLATION_Y in params):
        d2_tranformed = transform_image_2d(data2, [Transform.TRANSLATION_Y],
                                           [param_vals[params.index(Transform.TRANSLATION_Y)]])

        par_val_d[params.index(Transform.TRANSLATION_Y)] = 0.1* derivative_ce(
            np.asarray(data1).ravel(), np.asarray(d2_tranformed).ravel())

    return par_val_d


def powell(data1, data2, params, x0, tol=1e-6, maxiter=100):

    def obj_func(args):
        print("hallo")
        return loss(data1, data2, params, args)

    grad = np.gradient(obj_func, x0)
    fx = obj_func(x0)
    
    d = -grad
    
    i = 0
    
    while i < maxiter and np.linalg.norm(grad) > tol:
        x1 = x0 + d
        grad = np.gradient(obj_func, x1)
        fx1 = obj_func(x1)
        
        if fx1 < fx:
            x0 = x1
            fx = fx1
            
            d = -grad
        
        else:
            d = -grad
        

        print("Iter:", i+1, "best param",
              [x.name for x in params], ":", x0)
        print("gradient", grad)
        print("loss", fx1)
        
        i += 1
    
    return [i for i in x0]

def obj_function():
    pass

def pow_impl(data1, data2, params):

    par_vals = [-1]*len(params)
    width2, height2 = data2.size

    if (Transform.ROTATION in params):
        par_vals[params.index(Transform.ROTATION)] = round(
            random.uniform(-360, 360), 3)
    if (Transform.TRANSLATION_X in params):
        par_vals[params.index(Transform.TRANSLATION_X)] = round(
            random.uniform(-width2/8, width2/8), 3)
    if (Transform.TRANSLATION_Y in params):
        par_vals[params.index(Transform.TRANSLATION_Y)] = round(
            random.uniform(-height2/8, height2/8), 3)
            
    return powell(data1, data2, params, par_vals)

def findTransformation(data1, data2, params, faster=False):
    # result = pso(data1, data2, params, faster)
    # result = gd(data1, data2, params)
    result = pow_impl(data1, data2, params)
    return result
