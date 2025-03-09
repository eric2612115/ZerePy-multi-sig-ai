import logging
from src.action_handler import register_action

logger = logging.getLogger("actions.timetool_actions")

@register_action("timetool-generate")
def timetool_generate(agent, **kwargs):
    """Generate text using TimeTool's augmented Claude with tool capabilities"""
    agent.logger.info("\n🤖 GENERATING TEXT WITH TIMETOOL")
    try:
        result = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="generate-text",
            params=[
                kwargs.get('prompt'),
                kwargs.get('system_prompt', agent._construct_system_prompt()),
                kwargs.get('model', None)
            ]
        )
        agent.logger.info("✅ Text generation completed!")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Text generation failed: {str(e)}")
        return None

@register_action("get-current-time")
def get_current_time(agent, **kwargs):
    """Get the current time using TimeTool"""
    agent.logger.info("\n🕒 GETTING CURRENT TIME")
    try:
        result = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="get-time",
            params=[]
        )
        agent.logger.info(f"✅ Current time: {result}")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Failed to get current time: {str(e)}")
        return None

@register_action("get-current-date")
def get_current_date(agent, **kwargs):
    """Get the current date using TimeTool"""
    agent.logger.info("\n📅 GETTING CURRENT DATE")
    try:
        result = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="get-date",
            params=[]
        )
        agent.logger.info(f"✅ Current date: {result}")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Failed to get current date: {str(e)}")
        return None

@register_action("get-current-datetime")
def get_current_datetime(agent, **kwargs):
    """Get the current date and time using TimeTool"""
    agent.logger.info("\n⏰ GETTING CURRENT DATE AND TIME")
    try:
        result = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="get-datetime",
            params=[]
        )
        agent.logger.info(f"✅ Current date and time: {result}")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Failed to get current date and time: {str(e)}")
        return None

@register_action("get-crypto-price")
def get_crypto_price(agent, **kwargs):
    """Get the current price of a cryptocurrency symbol"""
    symbol = kwargs.get('symbol', 'BTCUSDT')
    agent.logger.info(f"\n💰 GETTING CURRENT PRICE FOR {symbol}")
    try:
        result = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="get-symbol-price",
            params=[symbol]
        )
        agent.logger.info(f"✅ Current price for {symbol}: {result}")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Failed to get current price: {str(e)}")
        return None