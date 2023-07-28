# Modified List
- alphagen_qlib
   1. stock_data_.py
   2. calculator_.py
- alphagen/data/expression.py
   1. import from _.py
   2. Feature class*


# TODO List
- OPTOOL 重构-> alphagen/data/expression_ocean.py，继承下面四类运算符，实现对应的运算符(用 torch 替换 numpy)
  - UnaryOperator: 这种类型的操作符只接受一个操作数。例如，绝对值，符号，对数，CSRank（交叉部分排名）等。

  - BinaryOperator: 这种类型的操作符接受两个操作数。例如，加法，减法，乘法，除法，幂，较大，较小等。

  - RollingOperator: 这种类型的操作符接受一个操作数和一个时间差（delta time），在给定时间窗口内对操作数进行滚动操作。例如，平均，求和，标准差，方差，偏度等。

  - PairRollingOperator: 这种类型的操作符接受两个操作数和一个时间差，对这两个操作数在给定时间窗口内进行滚动操作。