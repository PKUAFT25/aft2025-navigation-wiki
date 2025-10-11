# AFT招新分享 —— Python计算加速实践

## 前言

大家好！我们今天要分享的这些库，每一个背后都凝聚了开发者巨大的心血，因此短短几句话是无法将其完全讲透的。本次分享更多的是希望能为大家打开一扇窗，让大家了解到这些优秀工具的存在，以便在未来遇到性能瓶颈时，能够快速找到合适的加速方向。

### 为什么我们需要加速Python计算？

在深入探讨加速技术之前，我们先来思考两个基本问题。

#### 症结所在：为什么Python这么慢？

Python的“慢”主要源于其设计哲学，具体原因可以归结为以下几点：

- **解释执行**：作为一门解释型语言，Python代码在运行时由解释器**逐行解释**并执行，而非预先编译成高效的机器码。这个过程本身就带来了额外的运行时开销。
- **动态类型**：Python的变量类型在运行时才确定，这种灵活性使得解释器需要**在运行时进行类型检查和处理**，增加了计算负担。
- **全局解释器锁（GIL）**：在最主流的Python实现CPython中，全局解释器锁（GIL）的存在，限制了**在同一进程中，任意时刻只有一个线程能够执行Python字节码**。这使得Python的多线程在计算密集型任务上无法真正利用多核CPU的优势。
- **高层抽象**：Python提供了丰富的高层次抽象，如列表、字典等内置数据结构。这些工具虽然极大地提升了开发效率，但其底层实现的复杂性也意味着比C++或Java中的低级数据结构要慢。
- **自动内存管理**：Python的垃圾回收机制（GC）虽然将开发者从繁琐的内存管理中解放出来，但GC需要**定期检查并清理不再使用的内存**，这个过程会暂停程序的正常执行，从而影响整体性能。

#### 优势所在：我们为什么依然选择Python？

尽管存在性能短板，Python依然是当今最受欢迎的编程语言之一，这得益于它无与伦比的优势：

- **广泛的库和框架**：Python拥有一个庞大且成熟的生态系统，涵盖了从数据科学、机器学习到Web开发、自动化的几乎所有领域。开发者可以轻松地站在巨人的肩膀上，而不是一切从零开始。
- **强大的社区支持**：一个庞大而活跃的社区意味着海量的学习资源、详尽的文档和及时的帮助。无论你是初学者还是资深专家，都能在社区中找到归属感和解决方案。
- **快速开发与低门槛**：简洁的语法、动态类型和高层抽象共同造就了Python极高的开发效率，特别适合需要快速原型设计和迭代的项目。对于非科班出身的同学来说，Python也是进入编程世界最平缓的坡道之一。
- **数据科学与机器学习的霸主**：Python已成为数据科学和机器学习领域的“事实标准”。几乎所有顶级的库和框架，如NumPy, Pandas, TensorFlow, PyTorch等，都以Python为核心。

这些特性共同决定了Python的广泛应用。从职业发展角度看，在座的许多同学未来可能希望成为**量化研究员**。在量化研究中，我们常常需要处理海量的逐笔交易（Trades）和报价（Quotes）数据。在这种场景下，提升Python的计算速度，就等同于提升自己的工作效率。

接下来，我将从 **Cython, Numba, 并行计算, NumPy, Pandas, 以及内存管理**六个方面，为大家介绍一些实用的Python计算加速方向。

## 1. Cython：Python与C的无缝桥梁

如果你的性能瓶颈在于某个核心算法，并希望获得接近C语言的极致性能，那么可以考虑使用Cython。

Cython是一种将Python代码编译为C/C++代码的工具。它允许你在Python代码中**嵌入C语言级别的类型声明和函数调用**，从而在保持Python便利性的同时，获得显著的性能提升。Cython特别适用于计算密集型任务，并且它生成的模块可以与现有的Python代码和库无缝兼容。

## 2. Numba：JIT编译的魔力

接下来是Numba，一个能给你的数值计算代码带来惊喜的工具。

Numba是一个开源的即时（Just-In-Time, JIT）编译器，它通过LLVM编译器架构，在运行时将Python函数（尤其是操作NumPy数组的函数）编译成高效的机器码。

#### Numba的优势

- **JIT编译**：只需一个简单的`@jit`装饰器，Numba就能在函数首次被调用时对其进行编译和优化。
- **NumPy支持**：Numba对NumPy的数组操作有深度优化，能显著加速科学计算代码。
- **并行计算**：支持多线程并行，通过简单配置即可利用多核CPU。
- **GPU加速**：Numba还能将代码编译到GPU上运行，为大规模并行计算提供强大动力。
- **可移植性**：Numba编译的代码是平台独立的，无需修改即可在不同操作系统上运行。

#### 代码示例

Numba的使用非常简单，我们来看一个例子：

```python
from numba import jit
import numpy as np
import time

# 一个普通的Python函数
def sum_array(x):
    total = 0
    for i in range(x.size):
        total += x[i]
    return total

# 使用Numba JIT装饰器加速的函数
@jit(nopython=True)
def sum_array_numba(x):
    total = 0
    for i in range(x.size):
        total += x[i]
    return total

# 创建测试数据
x = np.arange(1000000)

# 计时比较
start = time.time()
sum_array(x)
print(f"普通函数时间: {time.time() - start:.6f} 秒")

# 第一次调用Numba函数会包含编译时间
start = time.time()
sum_array_numba(x)
print(f"Numba函数首次调用时间 (含编译): {time.time() - start:.6f} 秒")

# 第二次调用则会非常快
start = time.time()
sum_array_numba(x)
print(f"Numba函数后续调用时间: {time.time() - start:.6f} 秒")
```

#### 注意事项

尽管Numba非常强大，但使用时也需注意：

- **使用装饰器**：`@jit`是最常用的装饰器。推荐使用`@njit`，它等同于`@jit(nopython=True)`，会强制Numba使用纯机器码模式，如果遇到不支持的Python特性会直接报错，有助于写出更高性能的代码。
- **代码兼容性**：Numba只支持Python语言的一个子集。要获得最佳性能，应确保函数中只使用了Numba支持的特性。
- **数据类型**：Numba对数据类型高度敏感。请尽量使用NumPy数组，并确保其`dtype`是Numba支持的数值类型。
- **避免Python原生对象**：在加速函数中，应避免使用Python的原生列表、字典等对象，因为Numba对它们的支持有限，且会严重影响性能。
- **并行计算**：使用`@jit(parallel=True)`可启用并行模式，此时循环应使用`prange`替代`range`来实现并行迭代。

## 3. 并行计算：释放多核潜力

接下来我们谈谈一个更通用的编程概念——并行计算，主要包括多线程和多进程。

#### 多线程 (Multithreading)

多线程是指在同一个进程中创建多个线程，它们**共享同一个进程的内存空间**，可以方便地访问相同的数据。

- **优点**:
  - **低开销**: 线程的创建和销毁比进程轻量。
  - **共享内存**: 线程间通信和数据共享非常容易。
  - **高响应性**: 适用于I/O密集型任务（如网络请求、文件读写），因为当一个线程等待I/O时，其他线程可以继续执行，从而提高程序响应速度。
- **缺点**:
  - **GIL限制**: 在Python中，由于GIL的存在，多线程在CPU密集型任务上无法实现真正的并行。
  - **同步问题**: 共享数据时必须小心处理线程同步，以避免数据竞争和死锁。

我们通常较少使用`threading`处理计算任务，主要是因为GIL的存在。但在处理I/O问题时，`threading`依然效果显著，因为Python在执行I/O操作时会主动释放GIL。

##### 代码示例

`threading`模块提供了简单易用的API来管理线程。

```python
import threading
import time

def worker(num):
    """线程执行的任务"""
    print(f"线程 {num} 开始...")
    time.sleep(2)
    print(f"线程 {num} 结束。")

threads = []  
for i in range(5):
    # 创建线程
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    # 启动线程
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("所有线程执行完毕。")
```

#### 多进程 (Multiprocessing)

多进程是指创建一个程序的多个独立进程，**每个进程拥有自己独立的内存空间**，互不干扰。

- **优点**:
  - **无GIL限制**: 多进程可以完美绕过GIL，充分利用多核CPU进行并行计算。
  - **隔离性**: 进程间相互隔离，一个进程的崩溃不会影响其他进程。
- **缺点**:
  - **高开销**: 进程的创建和销毁比线程更重。
  - **通信复杂**: 进程间通信（IPC）需要通过特定机制（如队列、管道）进行，比线程共享内存要复杂。

##### 使用场景

- CPU密集型任务，如科学计算、数据处理等。
- 需要高可靠性和隔离性的任务。

##### 代码示例

创建进程主要有两种方式：逐个创建和使用进程池。

```python
# 示例1：逐个创建进程
import multiprocessing
import time

def worker(num):
    """进程执行的任务"""
    print(f"进程 {num} 开始...")
    time.sleep(2)
    print(f"进程 {num} 结束。")

if __name__ == "__main__":
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print("所有进程执行完毕。")

    
  
# 示例2：使用进程池（推荐）
import multiprocessing
import time

def worker(num):
    """进程池中工作进程执行的任务"""
    print(f"工作进程 {num} 开始...")
    time.sleep(1)
    print(f"工作进程 {num} 结束。")
    return num * num

if __name__ == "__main__":
    # 创建一个包含4个进程的进程池
    with multiprocessing.Pool(processes=4) as pool:
        # 使用 map 方法将任务分发给进程池
        # 它会自动处理任务分发和结果收集
        results = pool.map(worker, range(10))

    print("任务执行完毕。")
    print("结果:", results)
```

**提示**：频繁创建和销毁进程的开销很大。对于需要执行大量相似任务的场景，使用**进程池**是更高效的选择。

#### Joblib：更优雅的并行方案

如果你觉得手动管理线程和进程有些繁琐，我强烈推荐`joblib`库。它提供了`Parallel`和`delayed`两个核心函数，让并行化变得异常简单。

而且，由于是高度封装的库，`joblib`的稳定性和功能性都更强，能帮你避免很多奇怪的并发错误。

##### CPU密集型任务示例

```python
from joblib import Parallel, delayed
import time

# 定义一个计算密集型函数
def compute_square(n):
    time.sleep(0.1) # 模拟计算耗时
    return n * n

numbers = range(10)
start_time = time.time()

# n_jobs=-1 表示使用所有可用的CPU核心
# joblib默认使用多进程后端，适合CPU密集型任务
results = Parallel(n_jobs=-1)(delayed(compute_square)(i) for i in numbers)

end_time = time.time()

print("结果:", results)
print(f"耗时: {end_time - start_time:.4f} 秒")
```



##### I/O密集型任务示例

```python
from joblib import Parallel, delayed
import requests
import time

# 定义一个I/O密集型函数
def fetch_url(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code
    except requests.RequestException as e:
        return str(e)

urls = [
    "[https://www.python.org](https://www.python.org)",
    "[https://www.google.com](https://www.google.com)",
    "[https://www.github.com](https://www.github.com)",
    "[https://www.example.com](https://www.example.com)",
]
start_time = time.time()

# prefer="threads" 指定使用多线程后端，适合I/O密集型任务
statuses = Parallel(n_jobs=4, prefer="threads")(delayed(fetch_url)(url) for url in urls)

end_time = time.time()
print("状态码:", statuses)
print(f"耗时: {end_time - start_time:.4f} 秒")
```

#### 注意事项

- **后端选择**: `joblib`默认使用**多进程**（`loky`后端），适合CPU密集型任务。对于I/O密集型任务，可以通过`prefer="threads"`切换到**多线程**后端，避免进程创建和通信的开销。
- **内存开销**: 多进程模式下，每个子进程都会复制主进程的内存空间，这可能**占用大量内存**。
- **数据传输**: 进程间传递大量数据会产生显著开销。应尽量减少主进程与子进程之间的数据传输。
- **错误处理**: 并行代码中的错误可能**不会直接在主进程中显示**，需要特别注意错误处理和日志记录。

## 4. NumPy 加速：拥抱GPU

OK！通用加速方法就介绍到这里。接下来，我们来探讨如何为数据科学的核心库NumPy和Pandas加速。

#### CuPy: GPU上的NumPy

首先是CuPy，它是一个实现了NumPy兼容接口的GPU数组库。通过CuPy，你可以将在CPU上运行的NumPy代码，几乎无缝地迁移到NVIDIA GPU上执行，从而享受GPU带来的惊人加速。

- **API兼容**: CuPy的API与NumPy高度兼容，大多数情况下，你只需将`import numpy as np`替换为`import cupy as cp`即可。
- **性能飞跃**: 在处理大规模数组时，CuPy在许多操作上比NumPy快几十甚至上百倍。

[图：CuPy 与 NumPy 性能对比]

当然，前提是你的电脑上有一块支持CUDA的NVIDIA显卡。

## 5. Pandas 加速：从正确使用到替换方案

Pandas是数据处理与分析的瑞士军刀。但在讨论加速库之前，我想先强调一下**正确使用Pandas**的重要性，因为很多性能问题源于不恰当的用法。

### 高效使用Pandas：事半功倍的技巧

#### 1. 拥抱矢量化，告别循环

问题：使用.apply, .iterrows或for循环逐行处理DataFrame，性能极差。

解决方法：尽可能使用Pandas和NumPy的矢量化操作，它们在底层由高效的C代码实现。

```python
# 慢的方式：apply逐行操作
df['new_col'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# 快的方式：矢量化操作
df['new_col'] = df['a'] + df['b']
```

#### 2. 避免在循环中修改DataFrame

问题：在循环中反复使用pd.concat或df.append来增长DataFrame，会导致性能灾难，因为每次操作都会重新分配内存并复制数据。

解决方法：先将每一步的结果收集到一个Python列表中，最后一次性调用pd.concat或pd.DataFrame进行构建。

```python
# 慢的方式：循环中append
# final_df = pd.DataFrame()
# for item in list_of_data:
#     final_df = final_df.append(pd.DataFrame([item]), ignore_index=True)

# 快的方式：先收集，再构建
list_of_rows = [...] # 假设这里是你的数据
final_df = pd.DataFrame(list_of_rows)

# 对于合并多个DataFrame也是同理
# result = pd.concat(list_of_dataframes, ignore_index=True)
```

#### 3. 一次聚合，多次计算

问题：对同一个分组键执行多次groupby操作。

解决方法：使用.agg()方法，在一个groupby对象上同时执行多个聚合计算。

```python
# 慢的方式：多次groupby
result1 = df.groupby('key')['col1'].sum()
result2 = df.groupby('key')['col2'].mean()

# 快的方式：一次groupby + agg
result = df.groupby('key').agg(
    col1_sum=('col1', 'sum'),
    col2_mean=('col2', 'mean')
)
```

#### 4. 善用`category`类型

问题：当一个字符串列中包含大量重复值时（即低基数），使用默认的object类型会占用大量内存。

解决方法：将这类列转换为category类型。Pandas会用整数编码来存储这些值，大大减少内存占用。

```python
# 慢的方式：默认使用object类型
df['col'].memory_usage(deep=True)

# 快的方式：转换为category类型
df['col'] = df['col'].astype('category')
# df['col'].memory_usage(deep=True)
```



#### 5. 合理使用索引

问题：在没有索引的列上进行查找和过滤操作，Pandas需要进行全表扫描。

解决方法：如果频繁根据某列进行查找，可以将其设置为索引，以利用哈希或排序索引带来的速度提升。

```python
# 慢的方式：在普通列上过滤
# result = df[df['col'] == value]

# 快的方式：先设置索引再查找
df_indexed = df.set_index('col')
result = df_indexed.loc[value]
```

### Pandas的替代与增强库

当优化了代码写法后性能仍不满足需求时，可以考虑以下几个库。

#### Bottleneck

Bottleneck是一个使用Cython编写的库，旨在加速NumPy和Pandas中的部分函数。它提供了一系列高度优化的移动窗口函数（如`move_mean`, `move_sum`）和聚合函数（如`nansum`, `nanmean`），在处理含有`NaN`值的数组时尤其高效。

[图：Bottleneck 与 NumPy 性能对比]

从性能对比可以看出，Bottleneck在某些运算上可以比NumPy快上百倍，相比Pandas的等效操作更是如此。

#### Polars

Polars是一个用Rust编写的高性能数据处理库，正迅速成为Pandas的有力竞争者。它在Kaggle等数据科学竞赛中备受青睐，因为它能轻松处理超出Pandas内存限制的大型数据集。

- **核心优势**:
  - **并行计算**: Polars从设计之初就支持多核并行计算，能自动利用所有CPU核心。
  - **高效内存管理**: 基于Apache Arrow列式内存格式，支持零拷贝操作，极大降低了内存占用。
  - **惰性求值**: 拥有查询优化器，可以分析整个计算链条并找到最高效的执行路径。

虽然Polars的API与Pandas不完全相同，需要一定的学习成本，但其带来的性能回报是巨大的。

```python
import polars as pl

# Polars API 示例
df_pl = pl.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": ["cat", "dog", "cat", "mouse", "dog"]
})

# 链式操作：过滤、分组、聚合
result = (
    df_pl.filter(pl.col("a") > 2)
    .group_by("b")
    .agg(pl.col("a").sum().alias("sum_a"))
    .sort("sum_a", descending=True)
)
print(result)
```

#### 在Pandas中启用Numba引擎

对于一些内置函数，如滚动窗口计算（`.rolling()`），Pandas允许你指定`engine='numba'`来利用Numba进行JIT加速。

##### 性能对比示例

```python
import pandas as pd
import numpy as np
import bottleneck as bn

# 准备测试数据
s_test = pd.Series(np.random.randn(1_000_000))

# Numba JIT版本的滚动求平均
@jit(nopython=True)
def rolling_mean_jit(array, window):
    result = np.full(array.shape, np.nan)
    for i in range(window - 1, len(array)):
        result[i] = np.mean(array[i - window + 1:i + 1])
    return result

# %timeit s_test.rolling(10).mean()
# 75.3 ms ± 15.2 ms per loop

# %timeit s_test.rolling(10).mean(engine="numba", engine_kwargs={"parallel": True})
# 2.21 ms ± 169 µs per loop

# %timeit pd.DataFrame(rolling_mean_jit(s_test.values, 10))
# 312 ms ± 8.6 ms per loop

# %timeit pd.DataFrame(bn.move_mean(s_test, 10, axis=0))
# 3.11 ms ± 85.2 µs per loop
```

从这个对比中可以看到，Pandas结合Numba引擎取得了非常好的加速效果，与专门的Bottleneck库性能相当。

## 6. 数据存储与内存优化

最后，我们来谈谈两个经常被忽略但至关重要的方面：数据I/O和内存占用。

### 数据存储与读取：选择合适的格式

数据存储格式对读取速度和硬盘空间占用有显著影响。

- **CSV (逗号分隔值)**
  - **优点**: 纯文本，人类可读，通用性强。
  - **缺点**: **占用空间大**，**读取速度慢**，不支持数据类型信息。当数据量超过1GB时，强烈不建议使用。
- **Parquet**
  - **优点**: **列式存储**，支持高效压缩（如Snappy, Gzip），**读取速度快**（尤其适合只读取部分列的场景），支持复杂数据类型。是大数据生态系统的标准格式之一。
- **Feather**
  - **优点**: 基于Arrow格式，专为**高速数据读写**设计，速度极快。
  - **缺点**: 文件体积可能比Parquet大。
- **HDF5**
  - **优点**: 为存储大规模科学数据设计，支持高效切片读取，支持多种数据类型和多维数组。
- **Pickle**
  - **优点**: Python专用格式，能序列化几乎所有Python对象，速度快。
  - **缺点**:
    - **兼容性差**: 仅限Python使用。
    - **安全风险**: **绝对不要加载来自不受信任来源的pickle文件**，因为反序列化过程可能执行任意恶意代码。
    - **空间效率**: 通常不如Parquet等专用压缩格式。

> **安全提示**: 汇丰的金融科技同学未来会学习网络安全课程，郑老师（也是我们AFT的指导老师）会提到因不当使用pickle文件导致的安全案例。所以，请对来路不明的`.pkl`文件保持警惕！

### 内存优化：精打细算你的数据类型

在Pandas和NumPy中，为数据选择正确的类型是控制内存占用的最直接方法。

- **整数类型**: `int8`, `int16`, `int32`, `int64` (数字越大，表示范围越大，占用内存越多)。还有对应的无符号整数`uint`系列。
- **浮点类型**: `float16`, `float32`, `float64` (精度和内存占用依次增加)。
- **对象类型 (`object`)**: 通常用于存储字符串，内存开销最大。
- **分类类型 (`category`)**: 适用于低基数（重复值多）的列，能大幅节省内存。

可以发现，`float64`占用的内存是`float32`的两倍。如果业务场景对精度要求没那么高，将`float64`转换为`float32`就能让内存占用减半。

下面这个函数可以自动扫描DataFrame的每一列，并将其转换为最节省内存的合适类型。

```python
import pandas as pd
import numpy as np

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    遍历DataFrame的所有列，并根据其内容将数据类型更改为
    能节省最多内存的类型。
    """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object" and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in str(col_type):
            # 对于字符串列，如果唯一值比例较低，则转换为category
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    return df

# 示例
df_raw = pd.DataFrame({
    'int_col': np.random.randint(0, 100, size=1000),
    'float_col': np.random.rand(1000) * 100,
    'object_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000)
})
df_raw['int_col_64'] = df_raw['int_col'].astype(np.int64)
df_raw['float_col_64'] = df_raw['float_col'].astype(np.float64)


print("优化前内存使用情况：")
print(df_raw.memory_usage(deep=True).sum(), "bytes")
print(df_raw.info(memory_usage='deep'))


df_optimized = optimize_memory(df_raw.copy())
print("\n优化后内存使用情况：")
print(df_optimized.memory_usage(deep=True).sum(), "bytes")
print(df_optimized.info(memory_usage='deep'))
```

# 写在最后！

以上就是今天分享的python加速计算的主要内容了！其中可能涉及到大家并不常用的操作！

但是，更多地是想让这个操作变成大家的泛认知之内的内容。

这样大家在vibe编程时，能够将其很轻异地转化为正确的提示词。

在工作场景中的应用也更加得心应手一些

作为加速计算的轻量工具书是非常好用的！