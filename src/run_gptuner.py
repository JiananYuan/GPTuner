from configparser import ConfigParser
import argparse
import time
import os
from dbms.postgres import PgDBMS
from dbms.mysql import  MysqlDBMS
from config_recommender.coarse_stage import CoarseStage
from config_recommender.fine_stage import FineStage
from knowledge_handler.knowledge_preparation import KGPre
from knowledge_handler.knowledge_transformation import KGTrans
from space_optimizer.knob_selection import KnobSelection

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("test", type=str)
    parser.add_argument("timeout", type=int)
    parser.add_argument("-seed", type=int, default=1)
    args = parser.parse_args()
    print(args)
    time.sleep(2)
    config = ConfigParser()


    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")


    # Select target knobs, write your api_base and api_key
    dbms._connect("benchbase")
    knob_selection = KnobSelection(db=args.db, dbms=dbms, benchmark=args.test, api_base="https://api.xiaoai.plus/v1", api_key="sk-FxmvAoiLMVZcWURg9dCb78B071D2446a9d33C3153bE6C1D9", model="gpt-3.5-turbo")
    knob_selection.select_interdependent_all_knobs()


    # prepare tuning lake and structured knowledge
    target_knobs_path = "/home/yjn/Desktop/work-2/GPTuner/knowledge_collection/postgres/target_knobs.txt"
    # target_knobs_path = "/home/knob/revision/GPTuner/knowledge_collection/postgres/target_knobs.txt"
    with open(target_knobs_path, 'r') as file:
        lines = file.readlines()
        target_knobs = [line.strip() for line in lines]


    # write your api_base and api_key
    knowledge_pre = KGPre(db=args.db, api_base="https://api.xiaoai.plus/v1", api_key="sk-FxmvAoiLMVZcWURg9dCb78B071D2446a9d33C3153bE6C1D9", model="gpt-3.5-turbo")
    knowledge_trans = KGTrans(db=args.db, api_base="https://api.xiaoai.plus/v1", api_key="sk-FxmvAoiLMVZcWURg9dCb78B071D2446a9d33C3153bE6C1D9", model="gpt-3.5-turbo")
    for knob in target_knobs:
        knowledge_pre.pipeline(knob)
        knowledge_trans.pipeline(knob)

    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")
    
    # store the optimization results
    folder_path = "../optimization_results/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  

    gptuner_coarse = CoarseStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
    )

    gptuner_coarse.optimize(
        name = f"../optimization_results/{args.db}/coarse/", 
        trials_number=10, 
        initial_config_number=10)
    time.sleep(20)

    
    gptuner_fine = FineStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed
    )

    gptuner_fine.optimize(
        name = f"../optimization_results/{args.db}/fine/",
        trials_number=10 # history trials + new tirals
    )   

