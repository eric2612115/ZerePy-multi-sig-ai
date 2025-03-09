import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

logger = logging.getLogger("crypto_tools.token_security")


class SecurityAnalysisError(Exception):
    """Exception raised for errors in the security analysis process."""
    pass


# Common security issues in token contracts
SECURITY_ISSUES = {
    "mint_function": {
        "description": "Contract contains functions that can mint new tokens",
        "severity": "high",
        "impact": "Could lead to unlimited token supply and devaluation"
    },
    "hidden_owner": {
        "description": "Contract has hidden or obfuscated owner privileges",
        "severity": "high",
        "impact": "Owner could have backdoor control without transparency"
    },
    "proxy_contract": {
        "description": "Contract is a proxy that can be upgraded",
        "severity": "medium",
        "impact": "Contract functionality can change without token holders' consent"
    },
    "transfer_restrictions": {
        "description": "Contract can restrict transfers or blacklist addresses",
        "severity": "medium",
        "impact": "Token transfers could be frozen for certain addresses"
    },
    "high_fee": {
        "description": "Contract includes high transfer fees (>5%)",
        "severity": "medium",
        "impact": "High fees could significantly impact traders"
    },
    "no_liquidity_lock": {
        "description": "No liquidity locking mechanism found",
        "severity": "medium",
        "impact": "Liquidity could be removed causing price crash"
    },
    "centralized_ownership": {
        "description": "Token ownership is highly centralized",
        "severity": "medium",
        "impact": "Whale accounts could manipulate price"
    },
    "no_audit": {
        "description": "No external security audit found",
        "severity": "low",
        "impact": "Contract may not have been professionally reviewed"
    },
    "honeypot_potential": {
        "description": "Potential honeypot characteristics detected",
        "severity": "critical",
        "impact": "Investors may be unable to sell tokens"
    }
}


async def analyze_token_security(
        token_symbol: str,
        chain_id: str,
        token_address: str,
        third_party_client=None,
        cave_client=None
) -> Dict[str, Any]:
    """
    Analyze token security by checking for common issues.

    Args:
        token_symbol: Token symbol
        chain_id: Chain ID where the token is deployed
        token_address: Token contract address
        third_party_client: Optional ThirdPartyClient for security data
        cave_client: Optional CaveClient for on-chain analysis

    Returns:
        A dictionary containing security analysis results
    """
    logger.debug(f"Analyzing security for {token_symbol} ({token_address}) on chain {chain_id}")

    try:
        # Initialize results structure
        security_result = {
            "symbol": token_symbol,
            "address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "issues_found": [],
            "risk_score": 0,
            "risk_level": "unknown",
            "analysis_complete": False
        }

        # Try Third Party security client if available
        if third_party_client:
            try:
                security_data = await third_party_client.get_security(
                    original_token=token_symbol,
                    target_token=token_address,
                    chain=chain_id,
                    amount="1000"
                )

                if security_data:
                    security_result["third_party_analysis"] = security_data

                    # Extract issues if present in the response
                    if "Risks" in str(security_data):
                        # Process the response to extract security issues
                        detected_issues = []

                        # Check for common security concerns in the response
                        security_keywords = {
                            "mint": "mint_function",
                            "hidden owner": "hidden_owner",
                            "proxy": "proxy_contract",
                            "blacklist": "transfer_restrictions",
                            "high fee": "high_fee",
                            "liquidity lock": "no_liquidity_lock",
                            "centralized": "centralized_ownership",
                            "no audit": "no_audit",
                            "honeypot": "honeypot_potential"
                        }

                        response_text = str(security_data).lower()
                        for keyword, issue_type in security_keywords.items():
                            if keyword in response_text:
                                if "no " + keyword in response_text and issue_type not in ["no_audit",
                                                                                           "no_liquidity_lock"]:
                                    continue  # Skip if negated

                                detected_issues.append({
                                    "type": issue_type,
                                    "description": SECURITY_ISSUES[issue_type]["description"],
                                    "severity": SECURITY_ISSUES[issue_type]["severity"],
                                    "impact": SECURITY_ISSUES[issue_type]["impact"]
                                })

                        security_result["issues_found"] = detected_issues

                    # Try to extract security score
                    score_match = re.search(r"Security Score['\":]+\s*['\"]*(\d+)['\"]*", str(security_data))
                    if score_match:
                        try:
                            security_score = int(score_match.group(1))
                            security_result["risk_score"] = 10 - security_score  # Convert to risk score (10 - security)
                        except (ValueError, IndexError):
                            pass

            except Exception as e:
                logger.warning(f"Error getting third party security data: {str(e)}")

        # Try CARV on-chain analysis if available
        if cave_client:
            try:
                # Formulate a security query for the token
                security_query = f"Analyze the security risks of token {token_symbol} with contract address {token_address} on chain {chain_id}. Look for mint functions, hidden owners, proxy capabilities, and high fees."

                carv_analysis = await cave_client.fetch_on_chain_data_by_llm(security_query)

                if carv_analysis:
                    security_result["on_chain_analysis"] = carv_analysis

                    # Extract additional detected issues if not already found
                    # (Implementation would depend on CARV response format)
            except Exception as e:
                logger.warning(f"Error getting CARV security data: {str(e)}")

        # Calculate risk level based on issues found
        if "risk_score" not in security_result or security_result["risk_score"] == 0:
            # Calculate based on issues if no explicit score
            issue_severity_weights = {
                "critical": 10,
                "high": 7,
                "medium": 4,
                "low": 1
            }

            if security_result["issues_found"]:
                total_severity = sum(issue_severity_weights.get(issue["severity"], 0)
                                     for issue in security_result["issues_found"])
                security_result["risk_score"] = min(10, total_severity)
            else:
                security_result["risk_score"] = 0

        # Map risk score to risk level
        risk_score = security_result["risk_score"]
        if risk_score >= 8:
            security_result["risk_level"] = "critical"
        elif risk_score >= 6:
            security_result["risk_level"] = "high"
        elif risk_score >= 3:
            security_result["risk_level"] = "medium"
        elif risk_score > 0:
            security_result["risk_level"] = "low"
        else:
            security_result["risk_level"] = "minimal"

        security_result["analysis_complete"] = True
        return security_result

    except Exception as e:
        logger.error(f"Error analyzing token security: {str(e)}")
        raise SecurityAnalysisError(f"Failed to analyze token security: {str(e)}")


def security_score(security_result: Dict[str, Any]) -> int:
    """
    Extract a security score from a security analysis result.

    Args:
        security_result: Security analysis result from analyze_token_security

    Returns:
        Security score (0-10, higher is more secure)
    """
    if not security_result or not isinstance(security_result, dict):
        return 0

    if "risk_score" in security_result:
        # Convert risk score (0-10) to security score (10-0)
        return 10 - security_result["risk_score"]

    # Default to middle score if no data
    return 5


async def check_contract_issues(
        token_address: str,
        chain_id: str,
        cave_client=None
) -> List[Dict[str, Any]]:
    """
    Check for specific issues in a token contract.

    Args:
        token_address: Token contract address
        chain_id: Chain ID where the token is deployed
        cave_client: Optional CaveClient for on-chain analysis

    Returns:
        A list of detected issues
    """
    logger.debug(f"Checking contract issues for {token_address} on chain {chain_id}")

    try:
        detected_issues = []

        if cave_client:
            # Form a query to specifically check for contract issues
            query = f"""
            Analyze the contract at address {token_address} on chain {chain_id} and check for the following issues:
            1. Mint functions that could create new tokens
            2. Hidden owner privileges
            3. Proxy implementation that could be changed
            4. Transfer restrictions or blacklist functions
            5. High transaction fees
            6. Ability to pause or freeze transfers
            7. Honeypot characteristics
            """

            response = await cave_client.fetch_on_chain_data_by_llm(query)

            # Process the response to extract issues
            # This implementation depends on the structure of CARV's response

            # For demonstration, return a sample response format
            return detected_issues

        # Return empty list if no clients available
        return []

    except Exception as e:
        logger.error(f"Error checking contract issues: {str(e)}")
        return []