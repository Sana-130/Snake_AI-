import pygame
import random
import sys, time, math

pygame.init()
fps=10
fpsclock=pygame.time.Clock()
font = pygame.font.Font('freesansbold.ttf', 32)

screen_height=300
screen_width=300
sur_obj=pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")
White=(255,255,255)#(random.randint(0,255),random.randint(0,255),random.randint(0,255))
black=(0,0,0)
blue=(0,0,255)
colr=(100,10,90)
col=(50,50,30)

game_stop=False

body_cells=[]
foodx_y=[]

step = 10
snake_bod_w= 10
snake_bod_h= 10

ROWS = round(screen_height / snake_bod_w)-1
COLUMN =round(screen_width / snake_bod_w)-1



class snakeBody():
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.x_vel =step
        self.y_vel =0
        self.rect=pygame.rect.Rect(self.x, self.y, snake_bod_w, snake_bod_h)
        self.prev_pos=6
        self.curr_pos=6
        self.id = id #right
        self.counter=0
        self.curr_checkpoint=0
        self.finish=False
        self.color=White
       
    def draw(self):
        pygame.draw.rect(sur_obj, self.color, self.rect)

    def get_pos(self):
        self.col= self.rect.x// snake_bod_w
        self.row= self.rect.y// snake_bod_w

        return self.col, self.row

    def head_move(self):
        self.rect.x += self.x_vel
        self.rect.y +=self.y_vel 

        self.draw()

    def move(self):
        self.rect.x += self.x_vel
        self.rect.y +=self.y_vel   
     
        self.draw()

    def move_right(self):
        self.x_vel = step
        self.y_vel = 0
        self.curr_pos = 6
        

    def move_left(self):
        self.x_vel = -step
        self.y_vel = 0
        self.curr_pos = 8

    def move_up(self):
        self.y_vel = -step
        self.x_vel = 0
        self.curr_pos = 2
        
    def move_down(self):
        self.y_vel = step
        self.x_vel = 0
        self.curr_pos = 5
    
    def no_change(self):
        self.x_vel = self.x_vel
        self.y_vel = self.y_vel
        self.curr_pos = self.curr_pos

    def get_leadPos(self, direction):
        
        self.prev_pos=self.curr_pos

        if direction==5 :
            self.move_down()
      
        if direction==6 :
           self.move_right()

        if direction==8:
            self.move_left()
        
        if direction==2 :
            self.move_up()
        
class Game():
    def __init__(self):
        self.body_cells=[]#snakeBody((10)* snake_bod_w, 150, 0)]
        #self.head=self.body_cells[0]
        self.total_score=0
        self.game_stop=False
        self.foodx_y=[]
        self.last_time=time.time()
    
    def generate_snake(self, n):
        for i in range(n):
            body=snakeBody((10-i)* snake_bod_w, 150, i)
            self.body_cells.append(body)

    def add_body(self):
        
        last = self.body_cells[-1]
        if(last.curr_pos==6):
            new=snakeBody(last.rect.x- snake_bod_w, last.rect.y, last.id+1)
        if(last.curr_pos == 8):
            new=snakeBody(last.rect.x+ snake_bod_w , last.rect.y, last.id+1)
        if(last.curr_pos == 5):
            new=snakeBody(last.rect.x, last.rect.y-snake_bod_h, last.id+1)
        if(last.curr_pos == 2):
            new=snakeBody(last.rect.x, last.rect.y+snake_bod_h, last.id+1)
        self.body_cells.append(new)
        new.curr_pos=last.curr_pos
        new.x_vel=last.x_vel
        new.y_vel =last.y_vel
    
    def checkForEat(self):
        head=self.body_cells[0]

        foodc, foodr=self.food_pos()

        headc, headr= head.get_pos()
        if headc==foodc and headr==foodr:
            self.total_score+=1
            self.create_food()
            self.add_body()
    
    def crash(self, x, y):
        in_bod=False
        for body in self.body_cells[1:]:
            if x in range(body.rect.x, body.rect.x+snake_bod_w+1) and y in range(body.rect.y ,body.rect.y+ snake_bod_h+1):
                in_bod=True    
                return in_bod
        return in_bod
    
    def generate(self):
        x=random.randint(2, COLUMN-3)* snake_bod_w
        y=random.randint(2, ROWS-3)* snake_bod_w
        return x, y

    def draw_food(self):
        pygame.draw.rect(sur_obj, (0,255,0), (self.foodx_y[0], self.foodx_y[1], snake_bod_w, snake_bod_h))

    def create_food(self):
        x,y=[random.randint(2, COLUMN-3)* snake_bod_w, random.randint(2, ROWS-3)* snake_bod_w]
        while self.crash(x,y):
            x, y=self.generate()
        self.foodx_y=[x,y]

    def wait(self):
    
        now=time.time() 
        diff=round(now-self.last_time, 2)
        if diff >0.05:
            self.last_time=now
            return True
        else:
            self.last_time = now
            return False
    
    def set_up(self):

        sur_obj.fill(black)

        for body in body_cells:
            body.draw()

        pygame.draw.rect(sur_obj, (0,255,0), (self.foodx_y[0], self.foodx_y[1], snake_bod_w, snake_bod_h))

        text = font.render(f'Score {self.total_score}', True, blue)
        textRect = text.get_rect()
        textRect.center=(70, 20)
        sur_obj.blit(text, textRect)

    def set_game_stop(self):
        self.game_stop=True

    def check_collision(self):
        head=self.body_cells[0]
        c,r= head.get_pos()
        if c==-1 or r==-1 or c==30 or r==30:
            self.set_game_stop()
   

        for body in self.body_cells[2:]:
            bodc, bodr=body.get_pos()
            if c==bodc and r==bodr:
                self.set_game_stop()

    def reset(self):
        self.body_cells=[]
        self.generate_snake(3)
        self.game_stop=False

    def snake_move(self):
        head=self.body_cells[0]
        head.move()
        for body in self.body_cells[1:]: 
            body.get_leadPos(self.body_cells[body.id-1].prev_pos)
            body.move()

    def build(self):
        self.generate_snake(3)
        self.create_food()

    def food_pos(self):
        foodc, foodr=self.foodx_y[0]// snake_bod_w, self.foodx_y[1]//snake_bod_w
        return foodc, foodr
    
    def draw(self):
        sur_obj.fill(black)

        pygame.draw.rect(sur_obj, (0,255,0), (self.foodx_y[0], self.foodx_y[1], snake_bod_w, snake_bod_h))

        text = font.render(f'Score {self.total_score}', True, blue)
        textRect = text.get_rect()
        textRect.center=(70, 20)
        sur_obj.blit(text, textRect)

    def update(self):         
        pygame.display.update() 

    def set_fps(self):
        fpsclock.tick(fps)
    
    def get_dist(self):
        head=self.body_cells[0]
        col, row= head.get_pos()
        foodcol, foodrow= self.food_pos()

        dist = math.sqrt((row - foodrow)**2 + (col - foodcol)**2)
        return dist
    
    #Agent functions

    def calculate_reward(self , prev_dist):
        head=self.body_cells[0]
        reward=0
        foodcol, foodrow = self.food_pos()
        col, row = head.get_pos()

        if row==foodrow and col==foodcol:
            reward=10
            return reward
   
        if col==-1 or row==-1 or col==30 or row==30:
            reward=-10
   
            return reward
        
        if self.game_stop==True:
            reward=-10
            return reward

    
        new_dist= self.get_dist()
        if new_dist<=prev_dist:
            reward=1
        else:
            reward=-1

        '''
        new_dist= get_dist()
        ndiff = new_dist - prev_dist
        pdiff = prev_dist - new_dist
        
        
        if ndiff>=0.8:
            reward = -5
        if ndiff>=0.6 and ndiff<0.8:
            reward= -4
        if ndiff >=0.4 and ndiff<0.6:
            reward =-3
        if ndiff>=0.2 and ndiff<0.4:
            reward=-2
        if ndiff>=0 and ndiff<0.2:
            reward=-1

        if pdiff>=0.8:
            reward = 5
        if pdiff>=0.6 and pdiff<0.8:
            reward= 4
        if pdiff >=0.4 and pdiff<0.6:
            reward =3
        if pdiff>=0.2 and pdiff<0.4:
            reward=2
        if pdiff>=0 and pdiff<0.2:
            reward=1
        '''
        distance = abs(row - foodrow) + abs(col - foodcol)
        normalized_distance = distance / 59   
        reward = reward*((round(1 + 9 * (1 - normalized_distance), 2))/10) 

        return reward
    
    def execute(self, action):
        
        head=self.body_cells[0]
        
        head.prev_pos= head.curr_pos
      
        if action==0 and head.curr_pos!=2 :#and wait():
            head.move_down()
        
        if action==1 and  head.curr_pos!=8 :#and wait():
            head.move_right()
        
        if action==2 and head.curr_pos!=6 :#and wait():
            head.move_left()
            
        if action==3  and head.curr_pos!=5 :#and wait():
            head.move_up()

        dist=self.get_dist()
        
        self.snake_move()
        
        next_state=self.get_state()

        reward=self.calculate_reward(dist)
        
        self.check_collision()

        if self.game_stop==True:
            done=True
        else:
            done=False

        
        self.checkForEat()

        return next_state, reward, done
    
    def get_state(self):
        
        head=self.body_cells[0]
        states=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

        c, r=  head.get_pos()
        foodc, foodr=self.food_pos()

        #print(c,r)
        #foodleft, foodright, foodup, fooddown
        if r>foodr:
            states[0][2]=1
            if c<foodc:
                states[0][1]=1

                #[0,1,1,0] upright
                pass
            else:
                states[0][0]=1
                #[1,0,1,0] upleft
                pass
        elif r<foodr:
            states[0][3]=1
            if c<foodc:
                states[0][1]=1
                #[0,1,0,1] #downright
                pass
            else:
                states[0][0]=1
                #[1,0,0,1] #downleft
                pass
        elif r==foodr:
            if c>foodc:
                states[0][0]=1
                #[1,0,0,0] #left
                pass
            else:
                states[0][1]=1
                #[0,1,0,0] #right
                pass
        elif c==foodc:
            if r>foodr:
                states[0][2]=1
                #[0,0,1,0] up
                pass
            else:
                states[0][3]=1
                #[0,0,0,1] down
                pass
        

        for body in body_cells[2:]: 
            bodc, bodr=body.get_pos()
            if bodr==r and c-bodc==-1:#bodc>c and head.curr_pos!=body.curr_pos :#and head.curr_pos+body.curr_pos==7 :#(abs(head.curr_pos-body.curr_pos)!=6 and abs(body.curr_pos-head.curr_pos)!=4):
                #danger right, danger left, danger up, danger down
                #print("danger right")
                states[0][4]=1
            if bodr==r and c-bodc==1:#c>bodc and head.curr_pos!=body.curr_pos :#and head.curr_pos+body.curr_pos==7:#(abs(body.curr_pos-head.curr_pos)!=1 and abs(body.curr_pos-head.curr_pos)!=6) :
                #print("danger left")
                states[0][5]=1
            if bodc==c and r-bodc==1:#bodr>r and head.curr_pos!=body.curr_pos :#and head.curr_pos+body.curr_pos==14: #body up
                #print("danger up") 
                states[0][6]=1
            if bodc==c and r-bodc==-1:#r>bodr and head.curr_pos!=body.curr_pos :#and head.curr_pos+body.curr_pos==14:
                #print("danger down")
                states[0][7]=1
        #print(states[0][4], states[0][5], states[0][6], states[0][7] )
        if c==29:
            states[0][4]=1 #danger right
        if c==0:
            states[0][5]=1 # danger left
        if r==29:
            states[0][7]=1 #dabger down
        if r==0:
            states[0][6]=1 #danger up


        if head.curr_pos==6:
            states[0][8]=1
        elif head.curr_pos==8:
            states[0][9]=1
        elif head.curr_pos==5:
            states[0][10]=1
        elif head.curr_pos==2:
            states[0][11]=1

        
        
        states[0][12]=r/(ROWS-1)
        states[0][13]=c/(COLUMN-1)
        states[0][14]=foodc/(COLUMN-1)
        states[0][15]=foodr/(ROWS-1)

        
        return states
    
game=Game()
game.build()

'''
while True:
   head=game.body_cells[0]
   game.draw()

   if game.game_stop!=True:
        game.draw_food()
        game.snake_move()
        game.check_collision()
        game.checkForEat() 
        #dist1, dist2=distfromfood()
       
     
    
        for eve in pygame.event.get():
            if eve.type==pygame.QUIT:
                pygame.quit()
                sys.exit()

    
            if eve.type == pygame.KEYDOWN and game_stop!=True:
                if eve.key ==pygame.K_DOWN and head.curr_pos!=2 and game.wait(): #DL
                    head.move_down()
                if eve.key ==pygame.K_RIGHT and head.curr_pos!=8 and game.wait():
                    head.move_right()           
                if eve.key ==pygame.K_LEFT and head.curr_pos!=6 and game.wait():
                    head.move_left()
                if eve.key ==pygame.K_UP and head.curr_pos!=5 and game.wait():
                    head.move_up()

        

        pygame.display.update()
        fpsclock.tick(fps)
    #time.sleep(0.07)
'''