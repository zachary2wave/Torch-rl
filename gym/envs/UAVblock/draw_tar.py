import pyglet

class Viewer(pyglet.window.Window):

    def __init__(self, block = None):
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.window.Window(500, 500)
        self.batch = pyglet.graphics.Batch()
        if block is not None:
            for x, y, w, l in zip(self.block["startx"],self.block["starty"],self.block["w"],self.block["l"]):
                self.point = self.batch.add(
                    4, pyglet.gl.GL_QUADS, None,  # 4 corners
                    ('v2f', [x, y,  # x1, y1
                             x+w, y,  # x2, y2
                             x+w, y+l,  # x3, y3
                             x, y+l]),  # x4, y4
                    ('c3B', (86, 109, 249) * 4))  # color
        score_label = pyglet.text.Label(text="Score: 0", x=10, y=575)