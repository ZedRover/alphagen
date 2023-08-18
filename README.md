# TODO List
- [x] OPTOOL 重构-> alphagen/data/expression_ocean.py，继承下面四类运算符，实现对应的运算符(用 torch 替换 numpy)
- [x] New calculator, combine IC and q9
- [x] Finish train process using new opts and feature
- [ ] Task:Generate over 100 factors 
- [ ] Fix:Refactor StockData, use shared memory to optimize ram cost. Need to refactor StockData and evaluate(self,expression) in Feature class
- [ ] Fix:Migrate from torch to numpy
- [ ] Fix:Compatibility of operators with egret
