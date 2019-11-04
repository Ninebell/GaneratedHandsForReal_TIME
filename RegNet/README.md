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
    ```shell script
    C:\users\local\RegNet> python regnet_main.py --data YOUR_DATA_PATH
    ```
    >Data path must same with GANerated_Dataset. I recommend you to use original Data Set.  
                                                                                                                     
    - if you have pretrained model
    
    ```shell script
    C:\users\localRegNet> python regnet_main.py --data YOUR_DATA_PATH --model YOUR_MODEL_PATH
    ```

