from loguru import logger
import sys

# Clear the log file by opening it in write mode
with open("my_log.log", "w") as file:
    pass

# Remove all existing handlers
logger.remove()

# Adjust log level to warning or higher for terminal
logger.add(sys.stderr, level="WARNING")

# Add a file handler to save all logs
logger.add("my_log.log", level="DEBUG", rotation="100 MB")

# Log messages
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")

# Log exceptions
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("You tried to divide by zero.")

@logger.catch
def my_function():
    2 / 0

my_function()