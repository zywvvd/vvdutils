# 功能描述

该仓库主要实现对分类算法的评价，含ap、p@r、pr曲线、混淆矩阵。

# 支持范围

该仓库支持二分类（binary classification）、多类别分类（multiclass classification）和多标签分类（multilabel classification）的评价。

# 使用方法

以多标签分类为例，真实标签输入形式为labels=[0, 2, 1, ...]，预测值输入形式为predicts=[[0.3, 0.7, 0], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3], ...]。

* **计算各个类别的AP**

```python
ap_dict = ClassifierEvalMultilabel.compute_ap(labels, predicts)
print('AP res is: ', ap_dict)
```

结果如下：

AP res is:  {0: 0.9999, 1: 0.8848, 2: 0.8139, 3: 0.8662, 4: 0.9982, 5: 0.9412}

* **计算各类别p@r（需要指定最小recall值）**

```python
prec_dict = ClassifierEvalMultilabel.compute_p_at_r(labels, predicts, recall_thresh=0.995)
print('P at R res is: ', prec_dict)
```

结果如下：

P at R res is:  {0: 0.9985, 1: 0.3177, 2: 0.2225, 3: 0.4499, 4: 0.9487, 5: 0.0307}

* **画各个类别的pr曲线（需要指定pr曲线的存放路径）**

```python
ClassifierEvalMultilabel.draw_pr_curve(labels, predicts, output_dir='./pr_curve')
```

结果如下：

<center>
    <img src='./img/cls_0.png' width=30%>
    <img src='./img/cls_1.png' width=30%>
    <img src='./img/cls_2.png' width=30%>
    <img src='./img/cls_3.png' width=30%>
    <img src='./img/cls_4.png' width=30%>
    <img src='./img/cls_5.png' width=30%>
</center>


* **计算整体漏报率和误报率（需要指定ok类别的index和最大能容忍的漏报率）**

```python
min_score, res_dict, fn_index_list, fp_index_list = ClassifierEvalMultilabel.compute_fnr_and_fpr(labels, predicts, ok_ind=0, fnr_thresh=0.005, fail_study=True)
print('Loubao and wubao res is: ', res_dict)
```

结果如下：

Loubao and wubao res is:  {'fnr': 0.0045, 'fpr': 0.019}

注意这里的min_score, fn_index_list和fp_index_list要和fail_study=True配合使用，fail_study=False时，这些值都没有意义。


* **将各个类别的failure case（含漏报和误报）打印出来**

```python
cls_dict = {0:'ok', 1:'0', 2:'1', 3:'2', 4:'3', 5:'8'}
ClassifierEvalMultilabel.draw_failure_cases(ok_ind, img_path_list, labels, predicts, min_score, fn_index_list, fp_index_list, cls_dict, res_dir=None)
```
这里需要指定较多的参数，含上述compute_fnr_and_fpr得到的min_score, fn_index_list, fp_index_list，还需指定ok类别的index，图片路径列表和cls_dict（用于显示用）等，打印的failure case示例如下：

<center>
    <img src='./img/loubao_1.png' width=30%>
    <img src='./img/loubao_2.png' width=30%>
    <img src='./img/loubao_3.png' width=30%>
    <img src='./img/wubao_1.png' width=30%>
    <img src='./img/wubao_2.png' width=30%>
    <img src='./img/wubao_3.png' width=30%>
</center>


