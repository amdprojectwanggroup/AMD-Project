from own_package.hparam_opt import hparam_opt

hparam_opt(model_mode='SNN', loader_file='./excel/data_loader/gold_data_loader', total_run=50,
           instance_per_run=1, hparam_file='./excel/hparams_SNN_desc_all.xlsx')