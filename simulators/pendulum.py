# TODO: This is just reused not working old code which should result in a simple pendulum


import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# global variable
x_ = 0
line = 0


def animate(i):
    global x_
    line_x = np.zeros(2)
    line_y = np.zeros(2)
    numsteps = x_.shape[0]
    line_x[1] = x_.reshape(numsteps, -1)[i][0]
    line_y[1] = x_.reshape(numsteps, -1)[i][1]
    line.set_data(line_x, line_y)
    return line,

def init():
    line.set_data([], [])
    return line,

def main(args):
    global x_, line
    ndim_s = 2
    ndim_m = 1

    numsteps = args.numsteps
    mode = args.mode
    disturbance = args.disturbance
    disturbance_noise = args.disturbance_noise

    # system
    angle = np.random.uniform(- np.pi/8.0, + np.pi/8.0, size=(1,1))
    angleSpeed = np.ones_like(angle) * 0.0
    l = 1.5
    g = -0.01
    friction = 0.99
    motorTorque = 0.01

    # new measurement
    x[0][0] = np.sin(angle[0,0])
    x[1][0] = np.cos(angle[0,0])
    # print("x:", x)

    angleSpeed = motorTorque * np.reshape(y[0][0],(1,1))

    # friction
    angleSpeed *= friction

    # # gravity
    angleSpeed += np.cos(angle) * g

    # add disturbance after 1000 timesteps
    if(i % 1000 > 950 and disturbance):
        angleSpeed += 0.1

    if(disturbance_noise>0):
        angleSpeed += np.random.standard_normal(1) * disturbance_noise

    # calculate new position
    angle += angleSpeed

    if(angle > 2.0 * np.pi):
        angle -= 2.0 * np.pi
    if(angle < 0.0):
        angle += 2.0 * np.pi
    #angle = angleSpeed

def animate_pendulum()

    # animate the pendulum
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=numsteps, interval=20, blit=True)

    plt.show()

if __name__ == "__main__":
    animate_pendulum()
