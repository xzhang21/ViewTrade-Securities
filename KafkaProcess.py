import os
from multiprocessing import Process
import json
import pymongo
# =============================================================================
# from kafka import KafkaConsumer
# from kafka import KafkaProducer
# from kafka.errors import KafkaError
# import stockstats
# =============================================================================
from datetime import datetime
import numpy as np


class Algorithm(object):
    def __init__(self, **stock_info):
        self.stock_max_np = np.array(stock_info["stock_max"])
        self.stock_min_np = np.array(stock_info["stock_min"])
        self.stock_close_np = np.array(stock_info["stock_close"])
        self.stock_amount_np = np.array(stock_info["stock_amount"])

    def simple_moving_average(self):
        n = 10
        weight = np.ones(n)
        weight /= weight.sum()
        stock_sma = np.convolve(self.stock_close_np, weight, mode='valid')

        return stock_sma

    def expo_moving_average(self):
        n = 10
        weight = np.linspace(1, 0, n)
        weight = np.exp(weight)
        weight /= weight.sum()
        stock_ema = np.convolve(self.stock_close_np, weight, mode='valid')

        return stock_ema


class MongoDBConnect(object):
    def __init__(self, **configs):
        self.IPAddress = configs["IPAddress"]
        self.Port = configs["Port"]
        self.dbName = configs["dbName"]
        self.Collection = configs["Collection"]

    def mongodb_connect(self, symbol, start, end):
        try:
            client = pymongo.MongoClient(self.IPAddress, self.Port)
            db = client[self.dbName]
            table = db[self.Collection]
            print(start[0],start[1],start[2])

            start_date = datetime(int(start[0]), int(start[1]), int(start[2]))
            end_date = datetime(int(end[0]), int(end[1]), int(end[2]))
            
            stock_high = []
            stock_low = []
            stock_close = []
            stock_open = []
            stock_volume = []
            stock_time = []

            for info in table.find({"Symbol": symbol, "Time": {'$gte': start_date, '$lt': end_date}}):
                stock_high.append(info["High"])
                stock_low.append(info["Low"])
                stock_close.append(info["Close"])
                stock_open.append(info["Open"])
                stock_volume.append(info["Volume"])
                stock_time.append(info["Time"])

            return stock_high, stock_low, stock_close,stock_open, stock_volume,stock_time
        except:
            print("Get Database error")


class Producer(object):
    def __init__(self, **config):
        self.kafka_host = config["kafka_host"]
        self.kafka_port = config["kafka_port"]
        self.kafka_topic = config["kafka_topic"]
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(kafka_host=self.kafka_host, kafka_port=self.kafka_port))
        self.producer.DEFAULT_CONFIG["api_version_auto_timeout_ms"] = 20000

    def send_json(self, params):
        try:
            params_message = json.dumps(params)
            producer = self.producer
            producer.send(self.kafka_topic,params_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)


class Consumer(object):
    def __init__(self, **config):
        self.kafka_host = config["kafka_host"]
        self.kafka_port = config["kafka_port"]
        self.kafka_topic = config["kafka_topic"]
        # self.group_id = config["kafka_groupId"]
        # self.consumer = KafkaConsumer(self.kafka_topic, group_id=self.group_id, bootstrap_servers='{kafka_host}:
        # {kafka_port}'.format(kafka_host=self.kafka_host, kafka_port=self.kafka_port),auto_offset_reset='earliest')
        # self.consumer.DEFAULT_CONFIG[""] = 20000
        # self.consumer.DEFAULT_CONFIG[""] = 500
        self.consumer = KafkaConsumer(self.kafka_topic, bootstrap_servers='{kafka_host}:{kafka_port}'.format(kafka_host=self.kafka_host, kafka_port=self.kafka_port), auto_offset_reset="latest", api_version_auto_timeout_ms=20000, metadata_max_age_ms=500)

    def consume_data(self):
        try:
            for msg in self.consumer:
                yield msg
        except KeyboardInterrupt as e:
            print(e)

    def subscribe(self, pattern):
        self.consumer.subscribe(pattern=pattern)


class KafkaProcess(Process):
    def __init__(self, **config):
        super().__init__()
        self.kafka_host = config["kafka_host"]
        self.kafka_port = config["kafka_port"]
        self.kafka_topic = config["kafka_topic"]
        self.stockInfo = config["stock_info"]

    def run(self):
        mongo = MongoDBConnect(IPAddress="192.168.110.116", Port=27017, dbName="chart", Collection="Day")
        stock_max, stock_min, stock_close, stock_amount = mongo.mongodb_connect(self.stockInfo["symbol"], self.stockInfo["startDate"].split('-'), self.stockInfo["endDate"].split('-'))
        algorithm_cal = Algorithm(stock_max=stock_max, stock_min=stock_min, stock_close=stock_close, stock_amount=stock_amount)
        np_array_to_list = algorithm_cal.expo_moving_average().tolist()
        print(np_array_to_list, self.kafka_topic)
        producer = Producer(kafka_host=self.kafka_host, kafka_port=self.kafka_port, kafka_topic=self.kafka_topic)
        producer.send_json(json.dumps({"result": np_array_to_list}))


if __name__ == '__main__':
    consumer = Consumer(kafka_host="192.168.110.128", kafka_port=9092, kafka_topic="na", kafka_groupId="ALAN1")
    message = consumer.consume_data()

    consumer.subscribe("^Alan.*_req$")
    # consumer.subscribe("^test1")
    while True:
        for i in message:
            if i.value:
                print(i)
                stockInfo = eval(i.value.decode("utf-8"))
                p = KafkaProcess(kafka_host="192.168.110.128", kafka_port=9092, kafka_topic=i.topic.replace('_req', '_res'), stock_info=stockInfo)
                p.start()


