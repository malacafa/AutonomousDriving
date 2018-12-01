from DQN import dqn
from ENV import Env
import os
import time 
import rospy
from std_msgs.msg import Int32
from time import sleep

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"

rospy.init_node('pub')
pub = rospy.Publisher('car', Int32, queue_size=20)
scores = []
print()
print("run")
print()
if __name__=='__main__':
    global_step = 0
    agent = dqn(365, 5, 100000000, 0.001) # state size 365, output size 5
    env = Env()
    #r = rospy.Duration(1,0) 
    for e in range(1,3000):
        done=False
        state= env.reset()
        score=0
        print()
        print("run")
        print()
        for t in range(1,3000):
            print()
            print("run")
            print()
            action = agent.get_action(state)
            pub.publish(action) # robot run
            rospy.sleep(10.)

            new_state, reward, done = env.step(action)

            agent.save_data(state, action, reward, new_state, done)
 
            if e % 10 == 0:
                agent.save_model()
                print("model saved at e=",e)

            if t>=500:
                print("time out")
                done = True
                break

            if len(agent.memory) >= agent.train_start:
                if global_step <= 100:
                    agent.train_model(128)
                else:
                    agent.train_model(128,True)

            score += reward
            state = new_state
            global_step += 1

        if done:
            agent.copy_weights()
            scores.append(score)
            print("Global step:",global_step," Ep:.",e," score:",score," memory len:",len(agent.memory))
            k= raw_input("press any key to continue!")
            if k=='1111':
                goal_x=int(raw_input('goal x'))
                goal_y=int(raw_input('goal y'))
                env.goal_x= goal_x
                env.goal_y= goal_y
        
        if agent.epsilon > 0.05:
            agent.epsilon *= 0.05