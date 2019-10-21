# RegNet 
### What is main idea of RegNet?
My opinion is ProjLayer. It can connect 2D Rendered map and 3D position.
So this can make up for each deficiency.  

In 3D, they can not exactly detect the position. In 2D, the can not know hand's depth.
3D can know hand's depth, 2D know where they are in image.   

![Alt text](../image/regnet_model.PNG)

### What is problem?
- RegNet can not detect multi hands.
    - I don't know how to solve to this problem. But more study about deep learning and pose estimation.  
    **I WILL SOLVE**.
    - Will study OpenPose.

## Problems encountered When I work
- How to implement ProjLayer in keras?
    - Just broadcast all points to make gaussian heat-map.

- Not module found
    -  How to solve   
        ```
        C:\users\local> set PYTHONPATH='YOUR clone path'
        ```
    
## How to use
- How to train?

- How to detect
gk
