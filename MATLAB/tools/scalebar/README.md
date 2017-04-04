# Matlab SCALEBAR
---
**A Matlab Tool for makeup a SCALEBAR**

+ dragable 
+ resizeable
+ unit-support
+ menucommand-support  

![image_1aruj9usb1t3ptr218ebrk019s39.png-32.9kB][1]
## Prepare
```matlab
plot(sin(1:0.1:10));
obj = scalebar;
```
## Operate by GUI
**Right-Click on the SCALE-LINE**, set `[X or Y] Length`, `[X or Y] Unit`. 

![image_1arulfu95uu07on1hvd16hrmfm.png-3.7kB][2]

**Right-Click on the SCALE-Y-LABLE**, set `Rotate`.
![image_1arulmubfgbhrcunv51s4dir513.png-1.9kB][3]

## Operate by Command
It's amost equal to **Operate by GUI**.
```
obj.XLen = 15;              %X-Length, 10.
obj.XUnit = 'm';            %X-Unit, 'm'.
obj.Position = [55, -0.6];  %move the whole SCALE position.
obj.hTextX_Pos = [1,-0.06]; %move only the LABEL position
obj.Border = 'UL';          %'LL'(default), 'LR', 'UL', 'UR'

```


  [1]: http://static.zybuluo.com/chenxinfeng/jsb4vvuo2bpntm4wg2a6i71e/image_1aruj9usb1t3ptr218ebrk019s39.png
  [2]: http://static.zybuluo.com/chenxinfeng/glwhxwbqraou51wx0av64yf7/image_1arulfu95uu07on1hvd16hrmfm.png
  [3]: http://static.zybuluo.com/chenxinfeng/asjxl1cfsbmm8fjzdsaqqrvx/image_1arulmubfgbhrcunv51s4dir513.png