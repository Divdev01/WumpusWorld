# WumpusWorld
Designing an agent to navigate the Wumpus World environment that is mentioned in Russel and Norvig, Artificial Intelligence, A Morden Approach, 4th Edition.
Three implemention methods are used
  1. Naive Aproach - Agent chooses action randomly (not wise)
  2. Probabilistic Approach - Agent chooses action based on Probabilistic reasoning
  3. Deep Q Learning Approach - uses epsilon greedy approach to choose an action
 
## Environment
  * Performance measure - 
      * +1000 for climbing out of the cave with gold
      * -1000 for falling into a pit/being eaten by the Wumpus
      * -1 for each Action
      * -10 for using up the arrow
       
   
  * Environment
      * 4x4 grid of rooms in cave
      * Agent always starts bottom left grid facing right
      * Locations of gold and Wumpus choses randomly with uniform distribution (except the square where agent starts)
      * Each can be a pit with probability 0.2 (except the square where agent starts)
      
  * Actuators - gives below actions (Action.py file) 
      * Forward - move one step forwrd
      * TurnLeft - change direction to the left (90°)
      * TurnRight - change direction to the right (90°)
      * Shoot - shoot arrow in straight line where agent is facing(1 arrow available)
      * Grab - pick the gold 
      * Climb - climb out of th cave(done in the starting grid onl) 
  
      
  * Sensors - gives below percepts
      * stench - received in squares adjacent to the Wumpus square 
      * breeze - squares directly ad=jacent to pits
      * glitter - square where gold is 
      * bump - when agent walks into a wal
      * scream - when Wumpus got killed with arrow
      
