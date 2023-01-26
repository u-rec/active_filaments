import matplotlib
import numpy as np


class G:
    N = 5
    R = 5E-1
    ks = 1E2
    k = 1E2
    l0 = 1.0
    a = 1.0
    dt = 5E-3
    L_pol = (N-1) * l0
    ksi_p = 0.1
    kappa = ksi_p * a * L_pol


class Filament:
    def __init__(self):
        self.particles = np.matrix([np.arange(0, G.N * 2, 2), np.zeros(G.N)]).T

    def x(self):
        return self.particles.T[0]

    def y(self):
        return self.particles.T[1]

    @staticmethod
    def shift(a):
        return np.roll(a, 1, axis=0)

    @staticmethod
    def deshift(a):
        return np.roll(a, -1, axis=0)

    def shifted_x(self):
        return self.shift(self.x())

    def shifted_y(self):
        return self.shift(self.y())

    def bond_force(self):
        t = self.vectors()
        length = np.abs(t).sum(axis=1)
        length[length == 0] = 1
        forces = - (G.kappa * t) + G.kappa * G.l0 * t / length
        forces[0] = 0
        forces = forces - self.deshift(forces)
        return forces

    def vectors(self):
        t = self.particles - self.shift(self.particles)
        t[0, 0] = 0
        t[0, 1] = 0
        return t

    def norms(self):
        x = self.x()
        y = self.y()
        return np.sqrt(np.multiply(x, x) + np.multiply(y,y))

    @staticmethod
    def normalize(a):
        x = np.asarray(a).T[0]
        y = np.asarray(a).T[1]
        len = np.sqrt(np.multiply(x, x) + np.multiply(y,y)).T
        len[len == 0] = 1
        return np.asmatrix([x/len, y/len]).T


    def rotated(self):
        a = self.vectors().T
        b = np.copy(a[0])
        a[0] = -a[1]
        a[1] = b
        return self.normalize(a.T)


    def angles_cos(self):
        t = self.particles - self.shift(self.particles)
        lengths = np.sqrt(np.asarray(t.T[0]) ** 2 + np.asarray(t.T[1]) ** 2)
        cos = np.ndarray.flatten(np.asarray(np.sum(np.multiply(t, self.deshift(t)), axis=1).T) / np.multiply(lengths, np.roll(lengths, -1)))
        cos[0] = 1
        cos[-1] = 1
        return cos

    def angles(self):
        t = np.asarray((self.particles - self.shift(self.particles)).T)
        ts = np.asarray(self.deshift(t.T).T)
        angles = np.multiply(np.sign(np.multiply(t[0], ts[1]) - np.multiply(ts[0], t[1])), np.arccos(self.angles_cos()))
        return angles

    # normalized vector pointing direction of force acting on particle _before_ angle
    def direction_first_particle_angle_force(self):
        rot = self.rotated()
        return self.deshift(rot)

    # normalized vector pointing direction of force acting on particle _after_ angle
    def direction_third_particle_angle_force(self):
        rot = self.rotated()
        return rot

    def angle_force(self):
        force_third_x = - G.ks * self.shift(self.angles()) * np.asarray(self.direction_third_particle_angle_force()).T[0]
        force_third_y = - G.ks * self.shift(self.angles()) * np.asarray(self.direction_third_particle_angle_force()).T[1]
        force_first_x = - G.ks * self.deshift(self.angles()) * np.asarray(self.direction_first_particle_angle_force()).T[0]
        force_first_y = - G.ks * self.deshift(self.angles()) * np.asarray(self.direction_first_particle_angle_force()).T[1]
        force_second_x = - (self.shift(force_first_x) + self.deshift(force_third_x))
        force_second_x[0] = 0
        force_second_x[-1] = 0
        force_second_y = -(self.shift(force_first_y) + self.deshift(force_third_y))
        force_second_y[0] = 0
        force_second_y[-1] = 0
        return np.asarray([force_first_x + force_second_x + force_third_x, force_first_y + force_second_y + force_third_y]).T

    def ev_force(self):
        diff_x = np.ones((G.N, G.N)) * np.asarray(self.x()) - np.ones((G.N, G.N)) * np.asarray(self.x()).T
        diff_y = np.ones((G.N, G.N)) * np.asarray(self.y()) - np.ones((G.N, G.N)) * np.asarray(self.y()).T
        near = np.logical_and(diff_x ** 2 + diff_y ** 2 < 4 * (G.R**2), np.identity(G.N) == 0)

        force_x = np.zeros((G.N, G.N))
        force_y = np.zeros((G.N, G.N))
        np.putmask(force_x, near, - G.k * (diff_x)
        force_y[near] = -G.k * diff_y
        return



f = Filament()
f.particles[1,1] = 1
print(f.particles)
print(f.bond_force())
print(f.angles())
print(f.angle_force())
print(f.ev_force())

def main():
    return


