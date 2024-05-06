# TODO List
- [x] OPTOOL 重构-> alphagen/data/expression_ocean.py，继承下面四类运算符，实现对应的运算符(用 torch 替换 numpy)
- [x] New calculator, combine IC and q9
- [x] Finish train process using new opts and feature
- [x] Task:Generate over 100 factors 
- [ ] Fix:Refactor StockData, use shared memory to optimize ram cost. Need to refactor StockData and evaluate(self,expression) in Feature class
- [ ] Fix:Migrate from torch to numpy
- [ ] Fix:Compatibility of operators with egret


# NOTE
- FEATURE : 修改 *alphagen/data/expression.py* 中的 *Feature* 类的 *evaluate* 方法的最后一行，用于读取 feature 数据
- Y : 修改 *alphagen_ocean/calculator.py* 中的 *__init__* 方法读取 ret 数据
- DATE : 修改 *alphagen_ocean/stock_data.py* 中的 *_get_data* 方法读取 dates 数据