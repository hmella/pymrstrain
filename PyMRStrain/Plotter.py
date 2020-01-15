import matplotlib.pyplot as plt
from numpy import clip

# Slice plotter
class IndexTracker:
  def __init__(self, ax, X, Y):
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    self.ax = ax
    self.X = X
    self.Y = Y
    rows, cols, self.slices = X.shape
    self.ind = 0
    self.im_X = self.ax[0].imshow(self.X[:, :, self.ind], cmap=plt.get_cmap('Greys_r'),
                interpolation='nearest', vmin=0, vmax=2)
    self.im_Y = self.ax[1].imshow(self.Y[:, :, self.ind], cmap=plt.get_cmap('Greys_r'),
                interpolation='nearest', vmin=0, vmax=2)
    self.update()

  def onscroll(self, event):
    if event.button == 'up':
        self.ind = clip(self.ind + 1, 0, self.slices - 1)
    else:
        self.ind = clip(self.ind - 1, 0, self.slices - 1)
    self.update()

  def update(self):
    self.im_X.set_data(self.X[:, :, self.ind])
    self.im_Y.set_data(self.Y[:, :, self.ind])
    self.ax[0].set_title('x-direction (s%s)' % self.ind)
    self.ax[1].set_title('y-direction (s%s)' % self.ind)
    self.im_X.axes.figure.canvas.draw()
    self.im_Y.axes.figure.canvas.draw()
