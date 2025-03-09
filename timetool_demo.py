#!/usr/bin/env python3
"""
ZerePy TimeTool Demo - Integration with ZerePy Framework

This script demonstrates how to use the TimeTool connection within the ZerePy framework.
"""

import logging
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from src.agent import ZerePyAgent
from src.helpers import print_h_bar

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("timetool_demo")


def main():
    # Load the TimeToolAgent
    try:
        agent = ZerePyAgent("timetool_agent")
        logger.info(f"\n✅ Successfully loaded agent: {agent.name}")
    except Exception as e:
        logger.error(f"Error loading agent: {e}")
        return

    # Make sure the timetool connection is configured
    if not agent.connection_manager.connections["timetool"].is_configured():
        logger.info("\n⚠️ TimeTool connection is not configured.")
        agent.connection_manager.configure_connection("timetool")

    # Setup the agent's LLM provider
    agent._setup_llm_provider()

    # Setup prompt toolkit
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'info': 'ansigreen',
    })
    session = PromptSession(style=style)

    print_h_bar()
    logger.info("\n==== ZerePy TimeTool Demo ====")
    logger.info("This demo shows how Claude can use custom tools to get time information.")
    logger.info("Hint: Try asking 'What time is it?' or 'What's today's date?' or 'What's the price of BTCUSDT?'")
    logger.info("Type 'exit' to quit\n")
    print_h_bar()

    # Interactive loop
    while True:
        try:
            prompt_message = HTML(f'<prompt>{agent.name}</prompt> > ')
            user_input = session.prompt(prompt_message).strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                logger.info("\nGoodbye! 👋")
                break

            if not user_input:
                continue

            response = agent.connection_manager.perform_action(
                "timetool",
                "generate-text",
                [user_input, agent._construct_system_prompt()]  # 確保這裡有正確的 system_prompt
            )
            logger.info(f"\n{agent.name}: {response}\n")
            print_h_bar()

        except KeyboardInterrupt:
            continue
        except EOFError:
            logger.info("\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")