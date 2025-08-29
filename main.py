from Models.Model import Model
from data.OrderBook import OrderBookService
from data.FeatureEngineer import FeatureEngineering
from Trading.execution import Trader
import json
def TRAIN(COIN):
    Int_INTERVAL = 5
    INTERVAL = f"{Int_INTERVAL}m"  # 5-minute intervals
    BATCHES = 16
    SEQUENCES = 1
    DAYS_HISTORY = 30
    
    print("="*60)
    print("🚀 STARTING CRYPTOCURRENCY TRADING SYSTEM")
    print("="*60)
    print(f"📊 Coin: {COIN}")
    print(f"⏰ Interval: {INTERVAL}")
    print(f"📈 Sequence Length: {SEQUENCES}")
    print(f"📅 Historical Data: {DAYS_HISTORY} days")
    print("="*60)
    
    if True:        
        print(f"\n🔄 Step 1: Fetching historical market data for {Int_INTERVAL} minutes...")
        orderbook_service = OrderBookService(Int_INTERVAL, COIN, days=DAYS_HISTORY)
        print("✅ Historical data fetched successfully")
        fe = FeatureEngineering(COIN)
        fe.run()
        model = Model(COIN,INTERVAL,BATCHES,SEQUENCES)
        metrics = model.EvaluateAllModels()     
        return metrics
def evaluate_all_coins():
    Coin_performance = {}
    Coin_list = [
    "BTC",
    #  "ETH",
    # "XRP",
    # "SOL",
    # "AUCTION",
    # "DOGE",
    # "SHIB",
    # "XLM",
    # "TRX",
    # "WIF",
    # "INJ",
    # "PEPE"
    ]
    for Coin in Coin_list:
        Metrics = TRAIN(Coin)
        Coin_performance[Coin] = Metrics
    print(Coin_performance)

def main():
    """Main function to run the trading system"""
    
    # Configuration parameters
    COIN = "BTC"
    Int_INTERVAL = 5
    INTERVAL = f"{Int_INTERVAL}m"  # 5-minute intervals
    BATCHES = 16
    SEQUENCES = 1
    DAYS_HISTORY = 30
    
    print("="*60)
    print("🚀 STARTING CRYPTOCURRENCY TRADING SYSTEM")
    print("="*60)
    print(f"📊 Coin: {COIN}")
    print(f"⏰ Interval: {INTERVAL}")
    print(f"📈 Sequence Length: {SEQUENCES}")
    print(f"📅 Historical Data: {DAYS_HISTORY} days")
    print("="*60)
    
    if True:        # Step 1: Fetch OrderBook data (historical data)
        print(f"\n🔄 Step 1: Fetching historical market data for {Int_INTERVAL} minutes...")
        orderbook_service = OrderBookService(Int_INTERVAL, COIN, days=DAYS_HISTORY)
        print("✅ Historical data fetched successfully")
        fe = FeatureEngineering(COIN)
        fe.run()
        print("\n🔄 Step 2: Setting up feature engineering...")
        feature_path = f"Infrastructure/{COIN}_Features.json"
        
        if os.path.exists(feature_path):
            with open(feature_path,'r') as f:
                features = json.load(f)
        else:
            features = fe.SelectTopFeatures()
            with open(feature_path,'w') as f:
                json.dump(features,f)
        feature_data = fe.get_data()
        print(f"✅ Feature engineering complete")
        print(f"📋 Selected {features} top features")
        print(f"📊 Processed {len(feature_data)} historical data points")
        
        print("\n🔄 Step 3: Initializing trading system...")
        trader = Trader(
            coin=COIN,
            interval=INTERVAL,
            batches=BATCHES,
            sequences=SEQUENCES,
            features=features,
            feature_data=feature_data
        )
        print("✅ Trading system initialized")
        print("\n🚀 Step 4: Starting real-time trading...")
        print("⚠️  Press Ctrl+C to stop the system gracefully")
        print("="*60)
        trader.run_realtime_trading()

    

if __name__ == "__main__":
    evaluate_all_coins()