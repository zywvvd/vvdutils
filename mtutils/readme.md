# 数据管理工具

 操作各种json格式的数据管理工具，支持按自定义属性划分数据集，类别合并，数据集统计等功能。



## Setup

该工具可作为package使用，在你的代码文件夹下进行一下操作即可：

``` shell
git clone [GIT_URL.git] data_manager
```



## 设计逻辑

在data_manager.py里我们定义了一个类叫：DataManager

DataManager类维护两个和标注数据相关的变量：

1. **record_list**: 该变量是一个dict类型的列表（之后介绍），列表的元素记录这每个数据单元的属性；数据单元根据不同ML任务而定，可以是带框的图片，或者缺陷类型本身

2. **class_dict**: 该数据类型定义class_name和class_id的关系；该变量的数据类型不限



### callback函数

DataManager类通过callback函数来实现灵活的数据操作。callback函数的操作对象（输入参数）是一个存在record_list里的元素（即dict类对象），DataManager的类方法会遍历每一个record_list里的元素，结合callback的返回结果实现各种操作。详细介绍见后文。



### 示例：构建一个DataManager数据实例

```python
from data_manager import DataManager

class_dict = {'OK': 0, 'NG': 1}
record_list = [
    {'image_path': 'A/001.jpg', 'class_name': 'OK'},
    {'image_path': 'A/002.jpg', 'class_name': 'NG'},
    {'image_path': 'B/001.jpg', 'class_name': 'OK'},
]
data = DataManager(record_list=record_list, class_dict=class_dict)
```

该数据集包含3个records，每个record记录这一张图片的路径和该图片对应的类别标签；class_dict里有OK、NG两种类别。



## 常用功能介绍

以上面的例子为例，我们可以对```data```做如下操作：

### 打印数据集：dump()

``` python
data.dump()
```

该命令在控制台的输出如下：

``` shell
[1/3] sample >>
class_name: OK
image_path: A/001.jpg

[2/3] sample >>
class_name: NG
image_path: A/002.jpg

[3/3] sample >>
class_name: OK
image_path: B/001.jpg

Class Dict:
{'OK': 0, 'NG': 1}
```



### 克隆数据集：clone()

``` python
data_new = data.clone()
```

类方法clone()返回一个新的DataManager的实例，该实例与原实例data有相同的内容。



### 过滤数据集：filter()

``` python
def condition_callback(record):
    return 'A/' in record['image_path']
data_in_A_folder = data.filter(condition_callback)
```

该方法可以根据condition_callback设置的规则来过滤数据。以上代码挑选出所有在文件夹"A/"中的数据。返回的data_in_A_folder也是一个DataManager实例。



### 提取数据信息：extract_info()

``` python
def info_callback(record):
    return record['image_path']
image_path_list = data.extract_info(info_callback)
```

以上代码返回所有样本的图片地址列表。



### 统计频率：occurrence()

``` python
def key_callback(record):
    return record['class_name']
occurrence_dict = data.occurrence(key_callback)
```

以上代码统计所有类别的发生频率，返回的occurrence_dict是一个dict结构。

key_callback函数返回一个hashable常量（这个例子中是类别的字符串），**或者一个列表**（考虑到检测标注的例子，每个record表示一张图片，该图片上可能要好几个检测结果，我们可以通过key_callback输出一个类别列表来统计数据集中所有类别的发生频率）。



### 去重：unique()

``` python
def key_callback(record):
    return record['image_path']
data_uniq = data.unique(key_callback)
```

以上代码移除所有图片路径重复的record，返回一个新的DataManager实例，该实例中没有重复'image_path'的record。

和类方法occurrence()一样，unique()的参数key_callback函数也返回一个hashable常量（这个例子中是图片路径的字符串）。unique()把所有key_callback返回值一样的records看做同一个record，并随机把重复的record从record列表中移除。



### 数据集划分：split()

``` python
def groupID_callback(record):
    folder_name = record['image_path'].split('/')[0]  # i.e. 'A/0001.jpg' -> 'A'
    return folder_name
data1, data2 = data.split(num_or_ratio=1, groupID_callback=groupID_callback)
```

以上代码根据样本所在的文件夹随机划分成两份数据。其中groupID_callback返回一个hashable常量（这里是图片文件夹名字），split会先根据该常量/属性给所有records分组，然后随机把所有分组划分成2个数据集data1和data2（DataManager实例），其中data1包含num_or_ratio个record分组，data2包含剩下的分组。

其中num_or_ratio可以是一个整数也可以是一个0到1之间的浮点数。前者设置data1所分配到的分组个数，后者同样设置data1所能分到的分组个数的百分比。

如果不传入groupID_callback参数，则split直接划分record_list本身：

``` python
data1, data2 = data.split(num_or_ratio=0.33)
```



### 数据集合并：merge()

合并两个数据集的操作如下：

``` python
data = data1.merge(data2)
```
也可以用 '+' 号：
``` python
data = data1 + data2 + data3
```

两个数据集合并时，要求它们有同样的class_dict。



### 寻找交集：intersection()

可以根据回调函数key_callback定位两个数据集的交集，操作如下：

``` python
data_intersection = data1.intersection(data2, key_callback=lambda rec: rec['image_path'])
```

以上操作要求两个数据集有同样的class_dict。



### 寻找差集：difference()

和intersection()类似，该函数通过设置回调函数key_callback定位两个数据集的差集：

``` python
data_difference = data1.difference(data2, key_callback=lambda rec: rec['image_path'])
```

data_difference是data1的一个子集，它的record成员的图片路径与所有data2中record成员的图片路径都不同。

以上操作要求两个数据集有同样的class_dict。



### 三人行：zip()

![](assets/everytime3p.jpg)

模仿zip的行为，可提供不同的key_callback函数控制不同数据集间的共性：

``` python
for rec1, rec2, rec3 in data1.zip(data2, data3, key_callback=lambda rec['info']['uuid']):
	...
    # key_callback=lambda rec['info']['uuid'] 是默认值
```



### 划分数据集：batch() 和 chunk()

和split()类似，这两个函数把数据随机划分成N等分：

``` python
# 把数据划分成40等分（最后一份的size可能比前面的少）
for ds in data.chunk(40):
    print(ds)
 
# 把数据划分成N等分，每一份的size是1000（最后一份的size可能比前面的少）
for ds in data.batch(1000):
    print(ds)
```

### 保存数据集：save()

保存数据集到文件：

``` python
data.save('foobar.json')  # 保存成json格式
data.save('foobar.pkl')   # 保存成pickle格式
```

### 加载数据集：load()

可加载的json文件需要满足以下格式：

``` plain
{
    'class_dict': “任何格式的数据”，
    'record':[
    	“任何格式的数据1”，
        “任何格式的数据2”，
        ...
    ]
}
```

可加载的json文件是一个dict结构，需要包含两个键值：1）**class_dict**，和2）**record**。

其中class_dict可以是任意结构的数据，我们建议用一个dict来反应class_name和class_id的映射关系。

record必须是一个列表，列表中的每个元素可以是任意结构的数据。

加载文件的代码如下：

``` python
data = DataManager.load('foobar.json')  # 从json文件加载数据
data = DataManager.load('foobar.pkl')   # 从pickle文件加载数据
```

