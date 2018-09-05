# 2018-kuaishou_benchmark

前段时间参加了一下快手的用户兴趣建模大赛，贴上比赛时自己队伍最为接近最终模型的benchmark（不使用visual数据线上0.75）和最终模型图，模型并不复杂，在benchmark上稍作修改即可实现。    
  
未构造太多特征，使用特征如下：  
1.user_id  
2.inter_num:用户交互数目和  
3.relative_time当前sample的photo目前的time时间戳减去该photo最早出现的时间戳  
4.duration_time  
5.time  
6.age_max  
7.woman_num  
8.man_num  
9.facerate_mean  
10.text_mean/max(TF-IDF)  
代码略丑。  
环境要求：python3，tensorflow-gpu1.4.0,keras2.1.2  
赛题地址：https://www.kesci.com/home/competition/5ad306e633a98340e004f8d1  

Kesci上的一份数据分析EDA：https://www.kesci.com/home/project/5b27b37af110337467aeb904  


![image](https://github.com/WLiK/2018-kuaishou_benchmark/blob/master/model_img/model.jpg)
