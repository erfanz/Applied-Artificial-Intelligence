import pygame
from pygame import *



DEPTH = 32
FLAGS = 0
DELAY = 0
KEYBOARD = True
# in blocks, 25 x 20
class Graphics:
    def __init__(self, level):
        pygame.init()
        self.rows = 3 * len(level)
        self.cols = 2 + len(level[0])
        DISPLAY = (32* self.cols, 32* self.rows)
        self.screen = display.set_mode(DISPLAY, FLAGS, DEPTH)
        #display.set_caption("Use arrows to move!")
        
        self.bg = Surface((32,32))
        self.bg.convert()
        self.bg.fill(Color("#FFFFFF"))
        self.entities = pygame.sprite.LayeredUpdates()
        
        i = j = 0
        # build the level
        maps = [{'X':Graphics.ExitLadder,'H':Graphics.Ladder},
                {'X':Graphics.ExitLadder,'H':Graphics.Ladder,'S':Graphics.Player},
                {'XHP_S':Graphics.Platform, '123456789' : Graphics.Trapdoor}]
        tile_maps = []
        for map in maps:
            tile_map = {}
            for chars in map:
                for char in chars:
                    tile_map[char] = map[chars]
            tile_maps.append(tile_map)
        
        for row in level:
            for map in tile_maps:
                self.entities.add(Graphics.Platform(i, j))
                i += 1
                for char in row:
                    if char in map:
                        Entity = map[char]
                        e = Entity(i, j)
                        if Entity == Graphics.Player:
                            self.player = e
                        self.entities.add(e)
                    i += 1
                self.entities.add(Graphics.Platform(i, j))
                i = 0
                j += 1
        self.entities.move_to_front(self.player)
        #print player._Sprite__g
        #player.sprite.move_to_front()
    
            
            #for e in pygame.event.get():
            #    if e.type == QUIT: raise SystemExit, "Good bye"
            #    if KEYBOARD:
            #        if e.type == KEYDOWN and e.key == K_ESCAPE: 
            #            raise SystemExit, "Good bye"
            #        if e.type in (KEYDOWN, KEYUP) and e.key in (K_LSHIFT, K_RSHIFT):
            #            jump = e.type == KEYDOWN
            #        if e.type == KEYDOWN and e.key in (K_LEFT, K_RIGHT):
            #            player.side(e.key == K_LEFT, jump)
            #        if e.type == KEYDOWN and e.key == K_UP:
            #            #if player.location in ladders:
            #            player.climb()

    def move_player(self, state):
        new_x = 1 + state[1]
        new_y = 1 + 3 * state[0]
        old_x, old_y = self.player.location
        self.player.move(new_x - old_x, new_y - old_y)

    def update(self):
        for i in range(self.cols):
            for j in range(self.rows):
                self.screen.blit(self.bg, (i * 32, j * 32))
            
        self.entities.draw(self.screen)
        
        pygame.display.flip()
    
    class Entity(pygame.sprite.Sprite):
        def __init__(self):
            pygame.sprite.Sprite.__init__(self)
    
    class Player(Entity):
        def __init__(self, i, j):
            self.location = i, j
            x, y = 32 * i, 32 * j
            Graphics.Entity.__init__(self)
            self.onGround = False
            self.image = Surface((32, 32))
            self.image.convert()
            self.image.fill(Color("#FF0000"))
            self.rect = Rect(x, y, 32, 32)
        
        def move(self, delta_i, delta_j):
            i, j = self.location
            self.location = i + delta_i, j + delta_j
            self.rect.left += 32 * delta_i
            self.rect.top += 32 * delta_j
            time.wait(DELAY)
        
        def side(self, left, jump):
            delta_i = (-1) ** left * (1 + jump)
            self.move(delta_i,0)
        
        def climb(self):
            self.move(0,-3)
        
        def fall(self):
            self.move(0,3)
    
    class Platform(Entity):
        def __init__(self, i, j):
            self.location = i, j
            x, y = 32 * i, 32 * j
            Graphics.Entity.__init__(self)
            self.image = Surface((32, 32))
            self.image.convert()
            self.image.fill(Color("#000000"))
            self.rect = Rect(x, y, 32, 32)
    
    class Trapdoor(Entity):
        def __init__(self, i, j):
            self.location = i, j
            x, y = 32 * i, 32 * j
            Graphics.Entity.__init__(self)
            self.image = Surface((32, 32))
            self.image.convert()
            self.image.fill(Color("#00FF00"))
            self.rect = Rect(x, y, 32, 32)
    
    class Ladder(Entity):
        def __init__(self, i, j):
            self.location = i, j
            x, y = 32 * i, 32 * j
            Graphics.Entity.__init__(self)
            self.image = Surface((32, 32))
            self.image.convert()
            self.image.fill(Color("#0000FF"))
            self.rect = Rect(x, y, 32, 32)

    class ExitLadder(Entity):
        def __init__(self, i, j):
            self.location = i, j
            x, y = 32 * i, 32 * j
            Graphics.Entity.__init__(self)
            self.image = Surface((32, 32))
            self.image.convert()
            self.image.fill(Color("#800080"))
            self.rect = Rect(x, y, 32, 32)