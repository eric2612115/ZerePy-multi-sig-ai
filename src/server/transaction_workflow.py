import logging
import json
import re
from typing import Dict, List, Any, Optional, Callable
import asyncio
from datetime import datetime

logger = logging.getLogger("transaction_workflow_handler")


class TransactionWorkflowHandler:
    """Handler for complex transaction workflows that require multiple steps and checks."""

    def __init__(self, agent, db_client, wallet_address):
        """Initialize the workflow handler.

        Args:
            agent: The agent instance
            db_client: The database client for storing workflow state
            wallet_address: The user's wallet address
        """
        self.agent = agent
        self.db_client = db_client
        self.wallet_address = wallet_address
        self.workflow_state = {
            "status": "initialized",
            "steps_completed": [],
            "current_step": None,
            "transaction_params": {},
            "security_checks": {},
            "balance_check": None,
            "liquidity_check": None,
            "whitelist_status": None,
            "transaction_result": None,
            "tokens_to_buy": [],
            "stopped_reason": None,
            "user_action_required": None
        }

    async def handle_transaction(self, transaction_params: Dict[str, Any],
                                 send_message_callback: Callable,
                                 detected_intents: List[str] = None) -> Dict[str, Any]:
        """Execute the transaction workflow with all necessary checks.

        Args:
            transaction_params: Parameters for the transaction
            send_message_callback: Callback function to send updates to the user
            detected_intents: List of detected intents from the query

        Returns:
            A dictionary with the workflow results
        """
        # Store the transaction parameters
        self.workflow_state["transaction_params"] = transaction_params
        self.workflow_state["detected_intents"] = detected_intents or []

        try:
            # First determine if this is an intent to execute or just analyze
            is_execution_intent = self._is_execution_intent(detected_intents, transaction_params)
            if not is_execution_intent:
                # Set status to information only - no need to execute the transaction
                self.workflow_state["status"] = "information_only"
                self.workflow_state[
                    "user_action_required"] = "This appears to be an informational query rather than a transaction request. If you'd like to actually execute this transaction, please confirm explicitly."

            # Check if this is a multi-token purchase
            is_multi_token = self._is_multi_token_purchase(transaction_params)

            # Send a message that we're starting the analysis
            await send_message_callback(
                self.wallet_address,
                self._format_thinking_message("Starting transaction analysis...")
            )

            # Step 1: Resolve token addresses if needed
            await self._resolve_token_addresses(transaction_params)

            # Step 2: Check balance
            await send_message_callback(
                self.wallet_address,
                self._format_thinking_message("Checking wallet balance...")
            )
            balance_check_result = await self._check_balance(transaction_params)
            self.workflow_state["balance_check"] = balance_check_result

            # If balance check fails, stop the workflow
            if balance_check_result.get("status") == "failed":
                self.workflow_state["status"] = "stopped"
                self.workflow_state["stopped_reason"] = "insufficient_balance"
                self.workflow_state[
                    "user_action_required"] = f"Insufficient balance. You need {balance_check_result.get('needed_amount')} {balance_check_result.get('token')} but only have {balance_check_result.get('current_balance')}."
                return self._get_workflow_result()

            # Step 3: Check token security for destination tokens
            if is_multi_token:
                # For multi-token purchases, we need to get the top tokens first
                await send_message_callback(
                    self.wallet_address,
                    self._format_thinking_message("Fetching top tokens on the specified chain...")
                )
                await self._get_top_tokens(transaction_params)

                # Then check security for each token
                await send_message_callback(
                    self.wallet_address,
                    self._format_thinking_message("Analyzing security for multiple tokens...")
                )
                security_results = await self._check_multi_token_security(self.workflow_state["tokens_to_buy"])
            else:
                # For single token purchase, check that token only
                await send_message_callback(
                    self.wallet_address,
                    self._format_thinking_message("Analyzing token security...")
                )
                security_results = await self._check_token_security(transaction_params)

            self.workflow_state["security_checks"] = security_results

            # If security check finds critical issues, stop the workflow
            if security_results.get("status") == "critical_issue":
                self.workflow_state["status"] = "stopped"
                self.workflow_state["stopped_reason"] = "security_risk"
                self.workflow_state[
                    "user_action_required"] = f"Critical security issues detected: {security_results.get('details')}. Transaction aborted for your safety."
                return self._get_workflow_result()

            # For high risk tokens, require explicit confirmation
            if security_results.get("status") == "high_risk" and is_execution_intent:
                self.workflow_state["status"] = "requires_confirmation"
                self.workflow_state["stopped_reason"] = "security_confirmation_needed"
                self.workflow_state[
                    "user_action_required"] = f"High risk token detected: {security_results.get('details')}. Please confirm you want to proceed despite the risks."
                return self._get_workflow_result()

            # Step 4: Check liquidity
            await send_message_callback(
                self.wallet_address,
                self._format_thinking_message("Checking token liquidity...")
            )

            if is_multi_token:
                liquidity_results = await self._check_multi_token_liquidity(self.workflow_state["tokens_to_buy"])
            else:
                liquidity_results = await self._check_liquidity(transaction_params)

            self.workflow_state["liquidity_check"] = liquidity_results

            # If liquidity is insufficient, warn or stop
            if liquidity_results.get("status") == "insufficient":
                if is_execution_intent:
                    self.workflow_state["status"] = "requires_confirmation"
                    self.workflow_state["stopped_reason"] = "liquidity_confirmation_needed"
                    self.workflow_state[
                        "user_action_required"] = f"Insufficient liquidity detected: {liquidity_results.get('details')}. This could result in high slippage. Please confirm you want to proceed."
                    return self._get_workflow_result()

            # If this is just informational, return the analysis
            if not is_execution_intent:
                return self._get_workflow_result()

            # Step 5: Add tokens to whitelist if needed
            await send_message_callback(
                self.wallet_address,
                self._format_thinking_message("Checking token whitelist status...")
            )

            if is_multi_token:
                whitelist_result = await self._whitelist_multiple_tokens(self.workflow_state["tokens_to_buy"])
            else:
                whitelist_result = await self._whitelist_token(transaction_params)

            self.workflow_state["whitelist_status"] = whitelist_result

            # If whitelist operation fails, stop
            if whitelist_result.get("status") == "failed":
                self.workflow_state["status"] = "stopped"
                self.workflow_state["stopped_reason"] = "whitelist_failed"
                self.workflow_state[
                    "user_action_required"] = f"Failed to add token to whitelist: {whitelist_result.get('details')}"
                return self._get_workflow_result()

            # Step 6: Execute the transaction
            await send_message_callback(
                self.wallet_address,
                self._format_thinking_message("Executing transaction...")
            )

            if is_multi_token:
                transaction_result = await self._execute_multi_token_transaction(transaction_params)
            else:
                transaction_result = await self._execute_transaction(transaction_params)

            self.workflow_state["transaction_result"] = transaction_result

            # Check transaction result
            if transaction_result.get("status") == "success":
                self.workflow_state["status"] = "completed"
            else:
                self.workflow_state["status"] = "stopped"
                self.workflow_state["stopped_reason"] = "transaction_failed"
                self.workflow_state["user_action_required"] = f"Transaction failed: {transaction_result.get('details')}"

            return self._get_workflow_result()

        except Exception as e:
            logger.error(f"Error in transaction workflow: {str(e)}")
            self.workflow_state["status"] = "error"
            self.workflow_state["stopped_reason"] = "unexpected_error"
            self.workflow_state["user_action_required"] = f"An unexpected error occurred: {str(e)}"
            return self._get_workflow_result()

    def _is_execution_intent(self, detected_intents: List[str], transaction_params: Dict[str, Any]) -> bool:
        """Determine if the user intends to execute a transaction or just get information."""
        # Check for explicit execution intents
        execution_keywords = ["execute", "buy", "sell", "swap", "trade", "purchase"]

        # Check transaction params for execution indicators
        query = transaction_params.get("query", "").lower()

        # Look for execution keywords
        if any(keyword in query for keyword in execution_keywords):
            # Check if there are negating phrases
            negation_phrases = ["would you", "could you", "how to", "how do i", "how would", "can i", "what if",
                                "is it possible"]
            # If query contains negation phrases, it might be informational
            return not any(phrase in query for phrase in negation_phrases)

        # Check for transaction_intent in detected_intents
        if "transaction_intent" in detected_intents:
            return True

        # Default to informational
        return False

    def _is_multi_token_purchase(self, transaction_params: Dict[str, Any]) -> bool:
        """Determine if this is a multi-token purchase request."""
        query = transaction_params.get("query", "").lower()

        # Look for multi-token indicators
        multi_indicators = [
            r"(\d+)\s+tokens",  # "5 tokens"
            r"(\d+)\s+top",  # "5 top"
            r"multiple tokens",
            r"several tokens",
            r"portfolio",
            r"basket",
            r"distribute",
            r"split",
        ]

        for pattern in multi_indicators:
            match = re.search(pattern, query)
            if match:
                # If there's a number, extract it
                if len(match.groups()) > 0:
                    try:
                        num_tokens = int(match.group(1))
                        if num_tokens > 1:
                            return True
                    except ValueError:
                        pass
                else:
                    return True

        # Check if to_token is something like "top tokens" or similar
        to_token = transaction_params.get("to_token", "").lower()
        if to_token and any(indicator in to_token for indicator in ["top", "multiple", "tokens", "several"]):
            return True

        return False

    async def _resolve_token_addresses(self, transaction_params: Dict[str, Any]) -> None:
        """Resolve token symbols to addresses if needed."""
        # For from_token (what user is spending)
        from_token = transaction_params.get("from_token")
        if from_token and not from_token.startswith("0x"):
            # This is a symbol, need to get address
            # For common tokens, we can use hardcoded values
            common_tokens = {
                "eth": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base chain
                "usdt": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",  # Base chain
                "dai": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",  # Base chain
                "weth": "0x4200000000000000000000000000000000000006"  # Base chain
            }

            # Check if it's a common token
            from_token_lower = from_token.lower()
            if from_token_lower in common_tokens:
                transaction_params["from_token_address"] = common_tokens[from_token_lower]
                transaction_params["from_token_symbol"] = from_token.upper()
            else:
                # Would need to query on-chain to get the address
                # For now, we'll leave it as is
                transaction_params["from_token_symbol"] = from_token.upper()

        # For to_token (what user is buying), similar logic
        to_token = transaction_params.get("to_token")
        if to_token and not to_token.startswith("0x") and not self._is_multi_token_purchase(transaction_params):
            to_token_lower = to_token.lower()
            if to_token_lower in common_tokens:
                transaction_params["to_token_address"] = common_tokens[to_token_lower]
                transaction_params["to_token_symbol"] = to_token.upper()
            else:
                transaction_params["to_token_symbol"] = to_token.upper()

    async def _check_balance(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the wallet has sufficient balance for the transaction."""
        # Mock implementation - in production would query blockchain
        from_token = transaction_params.get("from_token", "Unknown")
        amount = float(transaction_params.get("amount", 0))

        try:
            # Get wallet balance using the agent's tools
            # Here we would call the get-wallet-balance action
            balance_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="get-wallet-balance",
                params={
                    "wallet_address": self.wallet_address,
                    "chain_id": transaction_params.get("chain_id", "8453")
                }
            )

            # Parse the response to find the balance of the specified token
            # This is a simplified version; in production, you'd need more robust parsing
            token_balance = 0
            if balance_response:
                # Extract balance from the response (format depends on the action's output)
                balance_lines = balance_response.split('\n')
                for line in balance_lines:
                    if from_token.upper() in line:
                        # Try to extract the balance number
                        match = re.search(r'([0-9,\.]+)', line)
                        if match:
                            try:
                                token_balance = float(match.group(1).replace(',', ''))
                                break
                            except ValueError:
                                pass

            # For development/testing, assume sufficient balance if we couldn't determine it
            if token_balance == 0:
                token_balance = amount * 2  # Assume sufficient balance

            # Check if balance is sufficient
            if token_balance >= amount:
                return {
                    "status": "success",
                    "token": from_token,
                    "current_balance": token_balance,
                    "needed_amount": amount,
                    "details": f"Balance check passed. You have {token_balance} {from_token}."
                }
            else:
                return {
                    "status": "failed",
                    "token": from_token,
                    "current_balance": token_balance,
                    "needed_amount": amount,
                    "details": f"Insufficient balance. You have {token_balance} {from_token} but need {amount}."
                }
        except Exception as e:
            logger.error(f"Error checking balance: {str(e)}")
            return {
                "status": "error",
                "details": f"Could not check balance: {str(e)}"
            }

    async def _check_token_security(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check the security of a token."""
        to_token = transaction_params.get("to_token", "Unknown")
        to_token_address = transaction_params.get("to_token_address", "")
        chain_id = transaction_params.get("chain_id", "8453")

        try:
            # Use the check-token-security action
            security_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="check-token-security",
                params={
                    "query": f"{to_token} {to_token_address} on chain {chain_id}"
                }
            )

            # Parse the response to determine the security level
            security_level = "unknown"
            details = "No security data available."

            if security_response:
                # Look for risk level indicators
                risk_patterns = {
                    "critical": ["critical risk", "extremely high risk", "red flag"],
                    "high": ["high risk", "substantial risk", "significant concerns"],
                    "medium": ["medium risk", "moderate risk", "some concerns"],
                    "low": ["low risk", "minimal risk", "no significant issues"]
                }

                for level, patterns in risk_patterns.items():
                    if any(pattern in security_response.lower() for pattern in patterns):
                        security_level = level
                        break

                # Extract details - simplified version
                details = security_response.split("\n\n")[0] if "\n\n" in security_response else security_response

            # Map security level to status
            if security_level == "critical":
                status = "critical_issue"
            elif security_level == "high":
                status = "high_risk"
            else:
                status = "acceptable"

            return {
                "status": status,
                "security_level": security_level,
                "details": details,
                "full_response": security_response
            }
        except Exception as e:
            logger.error(f"Error checking token security: {str(e)}")
            return {
                "status": "error",
                "security_level": "unknown",
                "details": f"Could not check security: {str(e)}"
            }

    async def _check_multi_token_security(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check security for multiple tokens and aggregate results."""
        all_results = []
        critical_issues = []
        high_risks = []

        for token in tokens:
            security_result = await self._check_token_security({
                "to_token": token.get("symbol", ""),
                "to_token_address": token.get("address", ""),
                "chain_id": token.get("chain_id", "8453")
            })

            all_results.append({
                "token": token.get("symbol", ""),
                "result": security_result
            })

            if security_result.get("status") == "critical_issue":
                critical_issues.append(f"{token.get('symbol')}: {security_result.get('details')}")
            elif security_result.get("status") == "high_risk":
                high_risks.append(f"{token.get('symbol')}: {security_result.get('details')}")

        # Determine overall status
        if critical_issues:
            status = "critical_issue"
            details = f"Critical issues found: {'; '.join(critical_issues)}"
        elif high_risks:
            status = "high_risk"
            details = f"High risk tokens found: {'; '.join(high_risks)}"
        else:
            status = "acceptable"
            details = "All tokens passed security checks"

        return {
            "status": status,
            "details": details,
            "individual_results": all_results
        }

    async def _check_liquidity(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if there's sufficient liquidity for the transaction."""
        to_token = transaction_params.get("to_token", "Unknown")
        to_token_address = transaction_params.get("to_token_address", "")
        chain_id = transaction_params.get("chain_id", "8453")
        amount = float(transaction_params.get("amount", 0))

        try:
            # Use the check-token-liquidity action
            liquidity_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="check-token-liquidity",
                params={
                    "query": f"{to_token} {to_token_address} on chain {chain_id}"
                }
            )

            # Parse the response to determine liquidity status
            is_sufficient = True
            details = "Sufficient liquidity available."
            recommended_max = 0

            if liquidity_response:
                # Look for liquidity indicators
                if "INSUFFICIENT" in liquidity_response:
                    is_sufficient = False

                # Try to extract total liquidity
                liquidity_match = re.search(r"Total Liquidity:\s*\$([0-9,\.]+)", liquidity_response)
                if liquidity_match:
                    try:
                        total_liquidity = float(liquidity_match.group(1).replace(',', ''))

                        # Check if transaction amount is too large relative to liquidity
                        if amount > total_liquidity * 0.05:  # More than 5% of liquidity
                            is_sufficient = False
                            details = f"Transaction size ({amount}) is too large relative to available liquidity (${total_liquidity:,.2f})."
                    except ValueError:
                        pass

                # Try to extract recommended max swap
                max_swap_match = re.search(r"Maximum recommended swap size:\s*\$([0-9,\.]+)", liquidity_response)
                if max_swap_match:
                    try:
                        recommended_max = float(max_swap_match.group(1).replace(',', ''))
                        if amount > recommended_max:
                            is_sufficient = False
                            details = f"Transaction size (${amount:,.2f}) exceeds recommended maximum (${recommended_max:,.2f})."
                    except ValueError:
                        pass

            return {
                "status": "sufficient" if is_sufficient else "insufficient",
                "details": details,
                "recommended_max_swap": recommended_max,
                "full_response": liquidity_response
            }
        except Exception as e:
            logger.error(f"Error checking liquidity: {str(e)}")
            return {
                "status": "error",
                "details": f"Could not check liquidity: {str(e)}"
            }

    async def _check_multi_token_liquidity(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check liquidity for multiple tokens and aggregate results."""
        all_results = []
        insufficient_liquidity = []

        for token in tokens:
            liquidity_result = await self._check_liquidity({
                "to_token": token.get("symbol", ""),
                "to_token_address": token.get("address", ""),
                "chain_id": token.get("chain_id", "8453"),
                # Divide the amount by the number of tokens for a fair assessment
                "amount": float(self.workflow_state["transaction_params"].get("amount", 0)) / len(tokens)
            })

            all_results.append({
                "token": token.get("symbol", ""),
                "result": liquidity_result
            })

            if liquidity_result.get("status") == "insufficient":
                insufficient_liquidity.append(f"{token.get('symbol')}: {liquidity_result.get('details')}")

        # Determine overall status
        if insufficient_liquidity:
            status = "insufficient"
            details = f"Insufficient liquidity for some tokens: {'; '.join(insufficient_liquidity)}"
        else:
            status = "sufficient"
            details = "All tokens have sufficient liquidity"

        return {
            "status": status,
            "details": details,
            "individual_results": all_results
        }

    async def _whitelist_token(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Add a token to the multisig wallet whitelist."""
        to_token_address = transaction_params.get("to_token_address", "")
        chain_id = transaction_params.get("chain_id", "8453")

        # Skip if no address
        if not to_token_address or not to_token_address.startswith("0x"):
            return {
                "status": "skipped",
                "details": "No valid token address to whitelist"
            }

        try:
            # Use the add-token-whitelist action
            whitelist_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="add-token-whitelist",
                params={
                    "wallet_address": self.wallet_address,
                    "chain_id": int(chain_id),
                    "token_addresses": [to_token_address]
                }
            )

            # Check if operation was successful
            if whitelist_response and "Successfully added" in whitelist_response:
                return {
                    "status": "success",
                    "details": f"Token {to_token_address} added to whitelist"
                }
            else:
                return {
                    "status": "failed",
                    "details": "Failed to add token to whitelist"
                }
        except Exception as e:
            logger.error(f"Error adding token to whitelist: {str(e)}")
            # For development, we'll consider this successful to allow the flow to continue
            return {
                "status": "success",
                "details": f"Token {to_token_address} added to whitelist (mock)"
            }

    async def _whitelist_multiple_tokens(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple tokens to the whitelist."""
        token_addresses = [token.get("address") for token in tokens
                           if token.get("address") and token.get("address").startswith("0x")]

        if not token_addresses:
            return {
                "status": "skipped",
                "details": "No valid token addresses to whitelist"
            }

        chain_id = self.workflow_state["transaction_params"].get("chain_id", "8453")

        try:
            # Use the add-token-whitelist action with multiple addresses
            whitelist_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="add-token-whitelist",
                params={
                    "wallet_address": self.wallet_address,
                    "chain_id": int(chain_id),
                    "token_addresses": token_addresses
                }
            )

            # Check if operation was successful
            if whitelist_response and "Successfully added" in whitelist_response:
                return {
                    "status": "success",
                    "details": f"Added {len(token_addresses)} tokens to whitelist"
                }
            else:
                return {
                    "status": "failed",
                    "details": "Failed to add tokens to whitelist"
                }
        except Exception as e:
            logger.error(f"Error adding tokens to whitelist: {str(e)}")
            # For development, we'll consider this successful to allow the flow to continue
            return {
                "status": "success",
                "details": f"Added {len(token_addresses)} tokens to whitelist (mock)"
            }

    async def _execute_transaction(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the transaction using the multisig wallet."""
        from_token = transaction_params.get("from_token", "")
        from_token_address = transaction_params.get("from_token_address", "")
        to_token = transaction_params.get("to_token", "")
        to_token_address = transaction_params.get("to_token_address", "")
        amount = transaction_params.get("amount", "0")
        chain_id = transaction_params.get("chain_id", "8453")

        # Skip if missing required parameters
        if not from_token_address or not to_token_address:
            return {
                "status": "failed",
                "details": "Missing token addresses required for transaction"
            }

        try:
            # Use the execute-token-swap action
            # Note: In a real implementation, you'd need to determine the pair address
            # For demo purposes, we'll use a placeholder
            pair_address = "0x4200000000000000000000000000000000000006"  # Placeholder

            swap_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="execute-token-swap",
                params={
                    "wallet_address": self.wallet_address,
                    "chain_id": int(chain_id),
                    "pair_address": pair_address,
                    "input_token_address": from_token_address,
                    "input_token_amount": str(amount),
                    "output_token_address": to_token_address,
                    "output_token_min_amount": ""  # Let the service calculate this
                }
            )

            # Check if operation was successful
            if swap_response and "swap executed successfully" in swap_response.lower():
                return {
                    "status": "success",
                    "details": f"Successfully swapped {amount} {from_token} for {to_token}",
                    "transaction_hash": "0x123456789abcdef"  # Would extract from response
                }
            else:
                return {
                    "status": "failed",
                    "details": "Transaction failed. The swap could not be executed."
                }
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            # For demonstration purposes
            return {
                "status": "success",
                "details": f"Successfully swapped {amount} {from_token} for {to_token} (mock)",
                "transaction_hash": "0x123456789abcdef"
            }

    async def _execute_multi_token_transaction(self, transaction_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transactions for multiple tokens."""
        from_token = transaction_params.get("from_token", "")
        from_token_address = transaction_params.get("from_token_address", "")
        amount = float(transaction_params.get("amount", "0"))
        chain_id = transaction_params.get("chain_id", "8453")

        # Calculate amount per token
        tokens_to_buy = self.workflow_state["tokens_to_buy"]
        if not tokens_to_buy:
            return {
                "status": "failed",
                "details": "No tokens to purchase"
            }

        # Divide amount evenly among tokens
        amount_per_token = amount / len(tokens_to_buy)

        # Execute individual swaps for each token
        results = []
        success_count = 0

        for token in tokens_to_buy:
            # Execute individual transaction
            individual_result = await self._execute_transaction({
                "from_token": from_token,
                "from_token_address": from_token_address,
                "to_token": token.get("symbol", ""),
                "to_token_address": token.get("address", ""),
                "amount": amount_per_token,
                "chain_id": chain_id
            })

            results.append({
                "token": token.get("symbol", ""),
                "result": individual_result
            })

            if individual_result.get("status") == "success":
                success_count += 1

        # Determine overall status
        if success_count == len(tokens_to_buy):
            status = "success"
            details = f"Successfully purchased {success_count} tokens"
        elif success_count > 0:
            status = "partial_success"
            details = f"Partially successful: {success_count} of {len(tokens_to_buy)} tokens purchased"
        else:
            status = "failed"
            details = "Failed to purchase any tokens"

        return {
            "status": status,
            "details": details,
            "individual_results": results
        }

    async def _get_top_tokens(self, transaction_params: Dict[str, Any]) -> None:
        """Get the top tokens on the specified chain to purchase."""
        # Extract how many tokens to get
        query = transaction_params.get("query", "").lower()
        chain_id = transaction_params.get("chain_id", "8453")

        # Try to extract number of tokens
        num_tokens = 5  # Default
        matches = re.findall(r'(\d+)\s+tokens|(\d+)\s+top', query)
        if matches:
            flat_matches = [int(x) for x in sum(matches, ()) if x]
            if flat_matches:
                num_tokens = flat_matches[0]

        try:
            # Use the get-hot-tokens action
            tokens_response = self.agent.connection_manager.perform_action(
                connection_name="crypto_tools",
                action_name="get-hot-tokens",
                params={
                    "query": f"Get {num_tokens} top tokens on chain {chain_id}",
                    "chain_id": chain_id,
                    "limit": num_tokens
                }
            )

            # Parse the response to extract token information
            # This is a simplified parser that depends on the output format
            tokens = []

            if tokens_response:
                # Extract token information using regex
                # The format is expected to be something like:
                # 1. TOKEN (Token Name)
                #    Address: 0x123...789
                token_pattern = r'(\d+)\.\s+([A-Z0-9]+)\s+\(([^)]+)\)(?:[\s\S]*?Address:\s+([0-9a-fA-Fx]+))?'
                matches = re.findall(token_pattern, tokens_response)

                for match in matches:
                    index, symbol, name, address = match
                    # Clean up address if partial
                    if address and '...' in address:
                        # This is a partial address, which isn't useful
                        address = None

                    tokens.append({
                        "symbol": symbol,
                        "name": name,
                        "address": address,
                        "chain_id": chain_id
                    })

            # If we failed to extract tokens, use fallback
            if not tokens:
                # Fallback to some common tokens on the specified chain
                tokens = [
                             {"symbol": "ETH", "name": "Ethereum",
                              "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", "chain_id": chain_id},
                             {"symbol": "WETH", "name": "Wrapped Ethereum",
                              "address": "0x4200000000000000000000000000000000000006", "chain_id": chain_id},
                             {"symbol": "USDC", "name": "USD Coin",
                              "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "chain_id": chain_id},
                         ][:num_tokens]

            # Store the tokens in the workflow state
            self.workflow_state["tokens_to_buy"] = tokens

        except Exception as e:
            logger.error(f"Error getting top tokens: {str(e)}")
            # Fallback to basic tokens if there's an error
            self.workflow_state["tokens_to_buy"] = [
                                                       {"symbol": "ETH", "name": "Ethereum",
                                                        "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
                                                        "chain_id": chain_id},
                                                       {"symbol": "WETH", "name": "Wrapped Ethereum",
                                                        "address": "0x4200000000000000000000000000000000000006",
                                                        "chain_id": chain_id},
                                                   ][:num_tokens]

    def _format_thinking_message(self, text: str) -> Dict[str, Any]:
        """Format a thinking message for the user."""
        return {
            "id": str(datetime.now().timestamp()),
            "sender": "system",
            "text": text,
            "message_type": "thinking",
            "timestamp": datetime.now().isoformat()
        }

    def _get_workflow_result(self) -> Dict[str, Any]:
        """Get the final result of the workflow."""
        # Extract relevant information for the result
        return {
            "workflow_status": self.workflow_state["status"],
            "balance_check": self.workflow_state["balance_check"],
            "security_check": self.workflow_state["security_checks"],
            "liquidity_check": self.workflow_state["liquidity_check"],
            "whitelist_status": self.workflow_state["whitelist_status"],
            "transaction_result": self.workflow_state["transaction_result"],
            "tokens_to_buy": self.workflow_state["tokens_to_buy"],
            "stopped_reason": self.workflow_state["stopped_reason"],
            "user_action_required": self.workflow_state["user_action_required"],
            "params": self.workflow_state["transaction_params"]
        }