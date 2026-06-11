from core.orchestrator import Crest

if __name__ == "__main__":
    bot = Crest()
    try:
        bot.start()
    except KeyboardInterrupt:
        print("Operator halted by user.")
