import matplotlib.pyplot as plt
from numpy import clip


# Slice plotter
class IndexTracker:
  def __init__(self, ax, X, Y, vrange=[]):
    if vrange == []:
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
    else:
        x_min, x_max = vrange
        y_min, y_max = vrange
    self.ax = ax
    self.X = X
    self.Y = Y
    rows, cols, self.slices = X.shape
    self.ind = 0
    self.im_X = self.ax[0].imshow(self.X[:, :, self.ind].T, cmap=plt.get_cmap('Greys_r'),
                interpolation='nearest', vmin=x_min, vmax=x_max)
    self.im_Y = self.ax[1].imshow(self.Y[:, :, self.ind].T, cmap=plt.get_cmap('Greys_r'),
                interpolation='nearest', vmin=y_min, vmax=y_max)
    self.im_X.axes.invert_yaxis()
    self.im_Y.axes.invert_yaxis()
    self.update()

  def onscroll(self, event):
    if event.button == 'up':
        self.ind = clip(self.ind + 1, 0, self.slices - 1)
    else:
        self.ind = clip(self.ind - 1, 0, self.slices - 1)
    self.update()

  def update(self):
    self.im_X.set_data(self.X[:, :, self.ind].T)
    self.im_X.axes.figure.canvas.draw()

    self.im_Y.set_data(self.Y[:, :, self.ind].T)
    self.im_Y.axes.figure.canvas.draw()
