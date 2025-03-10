from typing import List

import aiohttp, asyncio, json, logging, os
from dotenv import find_dotenv, load_dotenv


class WalletServiceClient:

    def __init__(self):
        load_dotenv(find_dotenv())
        self.base_url = os.getenv("WALLET_SERVICE_URL")
        self.header = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        self.session = None  # Initialize session to None

    async def _get_session(self):
        """Lazily creates an aiohttp ClientSession."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _request(self, method, url, data=None):
        """Centralized request handling with retries."""
        session = await self._get_session()
        try:
            if method.upper() == "GET":
                async with session.request(method, url, headers=self.header, params=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:

                print(f"Requesting {method} {url} with data: {data}")
                async with session.request(method, url, headers=self.header, json=data) as response:
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    return await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429 or e.status >= 500:  # Retry on rate limits or server errors
                logging.error(f"Request failed after multiple retries: {e}")
                raise  # Re-raise the exception after all retries fail
            else:
                logging.error(f"Request failed with client error: {e}")
                raise  # Re-raise for other client errors (4xx but not 429)
        except aiohttp.ClientConnectionError as e:
            logging.error(f"Connection error after multiple retries: {e}")
            raise
        except Exception as e:  # Catch any other exceptions
            logging.error(f"An unexpected error occurred: {e}")
            raise

    async def get_ai_wallet_address(self, wallet_address: str):
        """
        Retrieves an AI wallet by its ID.

        :param wallet_address: The ID of the wallet.
        :return: A dictionary containing the wallet information.
        """
        url = f"{self.base_url}/wallet/ai/{wallet_address}"
        return await self._request("GET", url)

    async def create_ai_wallet(self, owner_address: str):
        """
        Creates an AI wallet.

        :param owner_address: The address of the wallet owner.
        :return: A dictionary containing the wallet information.
        """
        url = f"{self.base_url}/wallet/create-ai"
        data = {"ownerAddress": owner_address}
        return await self._request("POST", url, data=data)

    async def create_multi_sig_wallet(self, owner_address: str, chain_id: int):
        """
        Creates a multi-sig wallet.

        :param owner_address: The address of the wallet owner.
        :param chain_id: The ID of the blockchain.
        :return: A dictionary containing the multi-sig wallet information.
        """
        url = f"{self.base_url}/wallet/create-wallet"
        data = {"ownerAddress": owner_address, "chain": chain_id}
        return await self._request("POST", url, data=data)

    async def get_multi_sig_wallet(self, wallet_address: str):
        """
        Retrieves a list of multi-sig wallets for a given owner address.

        :param wallet_address:  The owner address.
        :return: A dictionary containing the list of multi-sig wallets.
        """
        url = f"{self.base_url}/wallet/list?ownerAddress={wallet_address}"
        return await self._request("GET", url)

    async def get_multi_sig_wallet_whitelist(self, wallet_address: str, chain_id: int):
        """
        Gets the whitelist for a multi-sig wallet.

        :param wallet_address: The address of the multi-sig wallet.
        :param chain_id: The ID of the blockchain.
        :return:  The response from the whitelist endpoint.
        """
        url = f"{self.base_url}/wallet/whitelist"
        data = {"chain": str(chain_id), "safeWalletAddress": wallet_address}  # Ensure chain is string
        # Note: using json=data in _request already correctly handles the body.
        return await self._request("GET", url, data=data)

    async def add_multi_sig_wallet_whitelist(self, chain_id: int, safe_wallet_address: str, whitelist_signatures: list,
                                             token_addresses: list):
        """
        Adds tokens to the whitelist of a multi-sig wallet.
        """
        url = f"{self.base_url}/wallet/whitelist/add"
        data = {
            "chain": str(chain_id),  #Ensure chain is string
            "safeWalletAddress": safe_wallet_address,
            "whitelistSignatures": whitelist_signatures,
            "tokenAddresses": token_addresses
        }
        return await self._request("POST", url, data=data)

    async def remove_multi_sig_wallet_whitelist(self, chain_id: int, safe_wallet_address: str,
                                                whitelist_signatures: list, token_addresses: list):
        """
        Removes tokens from the whitelist of a multi-sig wallet.
        """
        url = f"{self.base_url}/wallet/whitelist/remove"
        data = {
            "chain": str(chain_id),  # Ensure chain is string
            "safeWalletAddress": safe_wallet_address,
            "whitelistSignatures": whitelist_signatures,
            "tokenAddresses": token_addresses
        }
        return await self._request("POST", url, data=data)

    async def multi_sig_wallet_swap(self, chain_id: int, ai_address: str, safe_wallet_address: str,
                                    input_token_address: str, input_token_amount: str, output_token_address: str,
                                    output_token_min_amount: str):
        """
        Performs a token swap through a multi-sig wallet.
        """
        url = f"{self.base_url}/wallet/swap"
        data = {
            "chain": str(chain_id),  #Ensure chain is string
            "aiAddress": ai_address,
            "safeWalletAddress": safe_wallet_address,
            "inputTokenAddress": input_token_address,
            "inputTokenAmount": input_token_amount,
            "outputTokenAddress": output_token_address,
            "outputTokenMinAmount": output_token_min_amount
        }
        return await self._request("POST", url, data=data)

    async def get_sonic_balance(self, wallet_address: str):
        """
        Retrieves the balance of a Sonic wallet.

        :param wallet_address: The address of the wallet.
        :return: A dictionary containing the wallet balance.
        """
        url = f"{self.base_url}/wallet/sonic/balance?owner={wallet_address}"
        return await self._request("GET", url)

    async def get_asset_info_list(self,chain_id: int, token_address: List[str] = None):
        """
        Retrieves the asset information list.

        :param chain_id: The ID of the blockchain.
        :param token_address: The address of the token.
        :return: A dictionary containing the asset information list.
        """
        url = f"{self.base_url}/asset/info/list"
        query = {"chainId": f"{chain_id}"}
        if token_address:
            query["tokenAddresses"] = token_address
        return await self._request("GET", url, data=query)

    async def get_asset_info(self, chain_id: int, token_address: str):
        """
        Retrieves the asset information.

        :param chain_id: The ID of the blockchain.
        :param token_address: The address of the token.
        :return: A dictionary containing the asset information.
        """
        url = f"{self.base_url}/asset/info"
        query = {"chainId": f"{chain_id}", "tokenAddress": [token_address]}
        return await self._request("GET", url, data=query)

    async def get_asset_price_list(self, chain_id: int, token_address: List[str] = None):
        """
        Retrieves the token price list.

        :param chain_id: The ID of the blockchain.
        :param token_address: The address of the token.
        :return: A dictionary containing the token price list.
        """
        url = f"{self.base_url}/asset/price/list"
        query = {"chainId": f"{chain_id}"}
        if token_address:
            query["tokenAddresses"] = token_address
        return await self._request("GET", url, data=query)

    async def get_asset_price(self, chain_id: int, token_address: str):
        """
        Retrieves the token price.

        :param chain_id: The ID of the blockchain.
        :param token_address: The address of the token.
        :return: A dictionary containing the token price.
        """
        url = f"{self.base_url}/asset/price"
        query = {"chainId": f"{chain_id}", "tokenAddress": token_address}
        return await self._request("GET", url, data=query)

    async def get_asset_info_with_price(self, chain_id: int, token_address: List[str] = None, token_name: List[str] = None):
        """
        Retrieves the asset information with price.

        :param chain_id: The ID of the blockchain.
        :param token_address: The address of the token.
        :param token_name: The name of the token.
        :return: A dictionary containing the asset information with price.
        """
        url = f"{self.base_url}/asset/info-price/list"
        query = {"chainId": f"{chain_id}"}
        res = await self._request("GET", url, data=query)
        if token_address:
            res = [i for i in res if i["tokenAddress"] in token_address]

        if token_name:
            res = [i for i in res if i["tokenName"] in token_name]

        return res

    """ ===== Wallet ====="""
    async def get_swap_history(self, wallet_address: str, chain_id: int):
        """
        Retrieves the swap history of a wallet.

        :param wallet_address: The address of the wallet.
        :param chain_id: The ID of the blockchain.
        :return: A dictionary containing the swap history.
        """
        url = f"{self.base_url}/wallet/swap/history"
        query = {"walletAddress": wallet_address, "chainId": f"{chain_id}"}
        return await self._request("GET", url, data=query)



    async def close(self):
        """Closes the aiohttp ClientSession."""
        if self.session:
            await self.session.close()


if __name__ == '__main__':
    async def main():
        client = WalletServiceClient()
        # print(await client.get_sonic_balance("0x2f500d4178d36ae358898b6ca398f2cfca0daff6"))
        # print(await client.get_sonic_balance("0x2f500d4178d36ae358898b6ca398f2cfca0daff6"))
        print(await client.add_multi_sig_wallet_whitelist(
            chain_id=146,
            safe_wallet_address="0x1234567890123456789012345678901234567890",
            whitelist_signatures=["0x2f500d4178d36ae358898b6ca398f2cfca0daff6"],
            token_addresses=["0x2f500d4178d36ae358898b6ca398f2cfca0daff6"]
        ))
        # print(await client.get_asset_info_list(146))

        # print(await client.get_asset_price_list(146))
        # print(await client.get_asset_info_with_price(146))


    asyncio.run(main())