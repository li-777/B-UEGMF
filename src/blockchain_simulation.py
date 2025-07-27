"""
Blockchain Simulation Environment
Simulates Ethereum blockchain network for testing smart contracts and access control
"""

import json
import time
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Transaction data structure"""
    tx_hash: str
    from_address: str
    to_address: str
    value: int  # Wei
    gas_price: int
    gas_limit: int
    gas_used: int
    data: str  # Contract call data
    timestamp: int
    block_number: Optional[int] = None
    status: str = 'pending'  # pending, mined, failed
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class Block:
    """Block data structure"""
    block_number: int
    block_hash: str
    parent_hash: str
    timestamp: int
    miner: str
    gas_limit: int
    gas_used: int
    transactions: List[Transaction]
    merkle_root: str
    nonce: int = 0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'transactions': [tx.to_dict() for tx in self.transactions]
        }


@dataclass
class Account:
    """Account data structure"""
    address: str
    private_key: str
    balance: int  # Wei
    nonce: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'balance': self.balance,
            'nonce': self.nonce
        }


class ContractCall:
    """Contract call data structure"""
    def __init__(self, contract_address: str, function_name: str, 
                 function_args: List[Any], from_address: str):
        self.contract_address = contract_address
        self.function_name = function_name
        self.function_args = function_args
        self.from_address = from_address
        self.timestamp = int(time.time())
        self.call_id = str(uuid.uuid4())


class Event:
    """Smart contract event"""
    def __init__(self, contract_address: str, event_name: str, 
                 event_data: Dict[str, Any], block_number: int):
        self.contract_address = contract_address
        self.event_name = event_name
        self.event_data = event_data
        self.block_number = block_number
        self.timestamp = int(time.time())
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict:
        return {
            'contract_address': self.contract_address,
            'event_name': self.event_name,
            'event_data': self.event_data,
            'block_number': self.block_number,
            'timestamp': self.timestamp,
            'event_id': self.event_id
        }


class MockIPFS:
    """Mock IPFS storage"""
    
    def __init__(self):
        self.storage: Dict[str, bytes] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add(self, data: bytes, filename: str = None) -> str:
        """Add data to IPFS, return CID"""
        # Generate mock CID
        cid = "Qm" + hashlib.sha256(data).hexdigest()[:44]
        
        self.storage[cid] = data
        self.metadata[cid] = {
            'filename': filename,
            'size': len(data),
            'timestamp': int(time.time())
        }
        
        logger.debug(f"File added to IPFS: {cid}")
        return cid
    
    def get(self, cid: str) -> Optional[bytes]:
        """Get data from IPFS"""
        return self.storage.get(cid)
    
    def pin(self, cid: str) -> bool:
        """Pin CID"""
        if cid in self.storage:
            logger.debug(f"CID pinned: {cid}")
            return True
        return False
    
    def get_metadata(self, cid: str) -> Optional[Dict]:
        """Get file metadata"""
        return self.metadata.get(cid)


class SimulatedBlockchain:
    """Simulated blockchain network"""
    
    def __init__(self, network_id: int = Config.Blockchain.NETWORK_ID):
        self.network_id = network_id
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.accounts: Dict[str, Account] = {}
        self.contracts: Dict[str, Any] = {}  # Contract address to contract instance mapping
        self.events: List[Event] = []
        
        # Mock IPFS
        self.ipfs = MockIPFS()
        
        # Network state
        self.gas_price = Config.Blockchain.GAS_PRICE
        self.block_time = 2  # seconds
        self.mining = False
        self.miner_thread = None
        
        # Create genesis block
        self._create_genesis_block()
        
        # Create default accounts
        self._create_default_accounts()
        
        logger.info(f"Blockchain network initialized - Network ID: {network_id}")
    
    def _create_genesis_block(self) -> None:
        """Create genesis block"""
        genesis_hash = hashlib.sha256(b"genesis_block").hexdigest()
        genesis_block = Block(
            block_number=0,
            block_hash=genesis_hash,
            parent_hash="0x0",
            timestamp=int(time.time()),
            miner="0x0",
            gas_limit=Config.Blockchain.GAS_LIMIT,
            gas_used=0,
            transactions=[],
            merkle_root="0x0"
        )
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def _create_default_accounts(self) -> None:
        """Create default accounts"""
        for i in range(Config.Blockchain.NUM_ACCOUNTS):
            private_key = hashlib.sha256(f"private_key_{i}".encode()).hexdigest()
            address = "0x" + hashlib.sha256(f"address_{i}".encode()).hexdigest()[:40]
            
            account = Account(
                address=address,
                private_key=private_key,
                balance=Config.Blockchain.INITIAL_BALANCE * 10**18  # Convert to Wei
            )
            
            self.accounts[address] = account
        
        logger.info(f"Created {len(self.accounts)} default accounts")
    
    def create_account(self, initial_balance: int = 0) -> Account:
        """Create new account"""
        account_id = str(uuid.uuid4())
        private_key = hashlib.sha256(account_id.encode()).hexdigest()
        address = "0x" + hashlib.sha256(f"addr_{account_id}".encode()).hexdigest()[:40]
        
        account = Account(
            address=address,
            private_key=private_key,
            balance=initial_balance
        )
        
        self.accounts[address] = account
        logger.info(f"New account created: {address}")
        return account
    
    def get_account(self, address: str) -> Optional[Account]:
        """Get account information"""
        return self.accounts.get(address)
    
    def get_balance(self, address: str) -> int:
        """Get account balance"""
        account = self.accounts.get(address)
        return account.balance if account else 0
    
    def transfer(self, from_address: str, to_address: str, 
                amount: int, gas_price: int = None) -> str:
        """Transfer funds"""
        gas_price = gas_price or self.gas_price
        gas_limit = 21000  # Standard transfer gas limit
        
        # Create transaction
        tx_data = f"{from_address}{to_address}{amount}{int(time.time())}"
        tx_hash = "0x" + hashlib.sha256(tx_data.encode()).hexdigest()
        
        transaction = Transaction(
            tx_hash=tx_hash,
            from_address=from_address,
            to_address=to_address,
            value=amount,
            gas_price=gas_price,
            gas_limit=gas_limit,
            gas_used=gas_limit,
            data="",
            timestamp=int(time.time())
        )
        
        # Validate and execute transaction
        if self._validate_transaction(transaction):
            self._execute_transaction(transaction)
            self.pending_transactions.append(transaction)
            logger.debug(f"Transfer transaction created: {tx_hash}")
            return tx_hash
        else:
            raise ValueError("Transaction validation failed")
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction"""
        from_account = self.accounts.get(transaction.from_address)
        if not from_account:
            logger.error(f"Sender account does not exist: {transaction.from_address}")
            return False
        
        total_cost = transaction.value + (transaction.gas_price * transaction.gas_limit)
        if from_account.balance < total_cost:
            logger.error(f"Insufficient balance: {from_account.balance} < {total_cost}")
            return False
        
        return True
    
    def _execute_transaction(self, transaction: Transaction) -> None:
        """Execute transaction"""
        from_account = self.accounts[transaction.from_address]
        to_account = self.accounts.get(transaction.to_address)
        
        # Deduct sender balance
        total_cost = transaction.value + (transaction.gas_price * transaction.gas_used)
        from_account.balance -= total_cost
        from_account.nonce += 1
        
        # Add to receiver balance
        if to_account:
            to_account.balance += transaction.value
        else:
            # Create new account if receiver doesn't exist
            new_account = Account(
                address=transaction.to_address,
                private_key="",
                balance=transaction.value
            )
            self.accounts[transaction.to_address] = new_account
        
        transaction.status = 'mined'
        logger.debug(f"Transaction executed: {transaction.tx_hash}")
    
    def deploy_contract(self, contract_code: str, from_address: str, 
                       constructor_args: List[Any] = None) -> str:
        """Deploy smart contract"""
        # Generate contract address
        contract_data = f"{from_address}{int(time.time())}{contract_code}"
        contract_address = "0x" + hashlib.sha256(contract_data.encode()).hexdigest()[:40]
        
        # Create deployment transaction
        tx_data = f"deploy{contract_address}{from_address}"
        tx_hash = "0x" + hashlib.sha256(tx_data.encode()).hexdigest()
        
        deploy_tx = Transaction(
            tx_hash=tx_hash,
            from_address=from_address,
            to_address="",  # Deployment transaction has empty to address
            value=0,
            gas_price=self.gas_price,
            gas_limit=3000000,  # Contract deployment requires more gas
            gas_used=2000000,
            data=contract_code,
            timestamp=int(time.time())
        )
        
        if self._validate_transaction(deploy_tx):
            self._execute_transaction(deploy_tx)
            self.pending_transactions.append(deploy_tx)
            
            # Store contract information
            self.contracts[contract_address] = {
                'code': contract_code,
                'constructor_args': constructor_args or [],
                'deploy_tx': tx_hash,
                'deployed_at': int(time.time())
            }
            
            logger.info(f"Contract deployed: {contract_address}")
            return contract_address
        else:
            raise ValueError("Contract deployment failed")
    
    def call_contract(self, contract_address: str, function_name: str,
                     function_args: List[Any], from_address: str) -> Any:
        """Call smart contract"""
        if contract_address not in self.contracts:
            raise ValueError(f"Contract does not exist: {contract_address}")
        
        # Create contract call
        call = ContractCall(contract_address, function_name, function_args, from_address)
        
        # Create transaction
        call_data = f"{function_name}({','.join(map(str, function_args))})"
        tx_data = f"{from_address}{contract_address}{call_data}{int(time.time())}"
        tx_hash = "0x" + hashlib.sha256(tx_data.encode()).hexdigest()
        
        call_tx = Transaction(
            tx_hash=tx_hash,
            from_address=from_address,
            to_address=contract_address,
            value=0,
            gas_price=self.gas_price,
            gas_limit=200000,
            gas_used=100000,
            data=call_data,
            timestamp=int(time.time())
        )
        
        if self._validate_transaction(call_tx):
            self._execute_transaction(call_tx)
            self.pending_transactions.append(call_tx)
            
            logger.debug(f"Contract call: {contract_address}.{function_name}")
            
            # Return simulated result
            return self._simulate_contract_call(call)
        else:
            raise ValueError("Contract call failed")
    
    def _simulate_contract_call(self, call: ContractCall) -> Any:
        """Simulate contract call execution"""
        # Return different results based on specific contract logic
        # For simplification, we return a successful result
        return {
            'status': 'success',
            'result': f"Function {call.function_name} executed successfully",
            'gas_used': 100000,
            'call_id': call.call_id
        }
    
    def emit_event(self, contract_address: str, event_name: str, 
                  event_data: Dict[str, Any]) -> None:
        """Emit smart contract event"""
        current_block = self.get_latest_block_number()
        event = Event(contract_address, event_name, event_data, current_block)
        self.events.append(event)
        logger.debug(f"Event emitted: {event_name} from {contract_address}")
    
    def get_events(self, contract_address: str = None, 
                  event_name: str = None) -> List[Event]:
        """Get event list"""
        filtered_events = self.events
        
        if contract_address:
            filtered_events = [e for e in filtered_events 
                             if e.contract_address == contract_address]
        
        if event_name:
            filtered_events = [e for e in filtered_events 
                             if e.event_name == event_name]
        
        return filtered_events
    
    def mine_block(self) -> Block:
        """Mine new block"""
        if not self.pending_transactions:
            return None
        
        # Get latest block
        latest_block = self.chain[-1]
        
        # Create new block
        new_block_number = latest_block.block_number + 1
        block_data = f"block_{new_block_number}_{int(time.time())}"
        block_hash = hashlib.sha256(block_data.encode()).hexdigest()
        
        # Calculate Merkle root
        tx_hashes = [tx.tx_hash for tx in self.pending_transactions]
        merkle_root = self._calculate_merkle_root(tx_hashes)
        
        # Calculate total gas used
        total_gas_used = sum(tx.gas_used for tx in self.pending_transactions)
        
        new_block = Block(
            block_number=new_block_number,
            block_hash=block_hash,
            parent_hash=latest_block.block_hash,
            timestamp=int(time.time()),
            miner=list(self.accounts.keys())[0],  # Use first account as miner
            gas_limit=Config.Blockchain.GAS_LIMIT,
            gas_used=total_gas_used,
            transactions=self.pending_transactions.copy(),
            merkle_root=merkle_root
        )
        
        # Update transaction status
        for tx in self.pending_transactions:
            tx.block_number = new_block_number
            tx.status = 'mined'
        
        # Add to chain
        self.chain.append(new_block)
        
        # Clear pending transactions
        self.pending_transactions.clear()
        
        logger.info(f"New block mined: #{new_block_number}, contains {len(new_block.transactions)} transactions")
        return new_block
    
    def _calculate_merkle_root(self, tx_hashes: List[str]) -> str:
        """Calculate Merkle tree root"""
        if not tx_hashes:
            return "0x0"
        
        if len(tx_hashes) == 1:
            return tx_hashes[0]
        
        # Simplified Merkle root calculation
        combined = "".join(tx_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def start_mining(self) -> None:
        """Start automatic mining"""
        if self.mining:
            return
        
        self.mining = True
        self.miner_thread = threading.Thread(target=self._mining_loop)
        self.miner_thread.daemon = True
        self.miner_thread.start()
        logger.info("Automatic mining started")
    
    def stop_mining(self) -> None:
        """Stop automatic mining"""
        self.mining = False
        if self.miner_thread:
            self.miner_thread.join()
        logger.info("Automatic mining stopped")
    
    def _mining_loop(self) -> None:
        """Mining loop"""
        while self.mining:
            time.sleep(self.block_time)
            if self.pending_transactions:
                self.mine_block()
    
    def get_latest_block_number(self) -> int:
        """Get latest block number"""
        return len(self.chain) - 1
    
    def get_block(self, block_number: int) -> Optional[Block]:
        """Get specified block"""
        if 0 <= block_number < len(self.chain):
            return self.chain[block_number]
        return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction information"""
        # Search transactions in all blocks
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_hash == tx_hash:
                    return tx
        
        # Search pending transactions
        for tx in self.pending_transactions:
            if tx.tx_hash == tx_hash:
                return tx
        
        return None
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        return {
            'network_id': self.network_id,
            'latest_block_number': self.get_latest_block_number(),
            'total_accounts': len(self.accounts),
            'total_contracts': len(self.contracts),
            'pending_transactions': len(self.pending_transactions),
            'gas_price': self.gas_price,
            'mining': self.mining,
            'block_time': self.block_time
        }
    
    def save_state(self, filepath: str) -> None:
        """Save blockchain state"""
        state = {
            'network_id': self.network_id,
            'chain': [block.to_dict() for block in self.chain],
            'accounts': {addr: acc.to_dict() for addr, acc in self.accounts.items()},
            'contracts': self.contracts,
            'events': [event.to_dict() for event in self.events],
            'gas_price': self.gas_price
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Blockchain state saved: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load blockchain state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.network_id = state['network_id']
            self.gas_price = state['gas_price']
            self.contracts = state['contracts']
            
            # Rebuild blockchain
            self.chain = []
            for block_data in state['chain']:
                transactions = [
                    Transaction(**tx_data) for tx_data in block_data['transactions']
                ]
                block_data['transactions'] = transactions
                self.chain.append(Block(**block_data))
            
            # Rebuild accounts
            self.accounts = {}
            for addr, acc_data in state['accounts'].items():
                self.accounts[addr] = Account(**acc_data)
            
            # Rebuild events
            self.events = []
            for event_data in state['events']:
                event = Event(
                    event_data['contract_address'],
                    event_data['event_name'],
                    event_data['event_data'],
                    event_data['block_number']
                )
                event.timestamp = event_data['timestamp']
                event.event_id = event_data['event_id']
                self.events.append(event)
            
            logger.info(f"Blockchain state loaded: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load blockchain state: {e}")
            raise


if __name__ == "__main__":
    # Test blockchain simulation environment
    logging.basicConfig(level=logging.INFO)
    
    print("Testing blockchain simulation environment...")
    
    # Create blockchain instance
    blockchain = SimulatedBlockchain()
    
    # Get accounts
    accounts = list(blockchain.accounts.values())
    alice = accounts[0]
    bob = accounts[1]
    
    print(f"Alice address: {alice.address}")
    print(f"Alice balance: {alice.balance / 10**18} ETH")
    print(f"Bob address: {bob.address}")
    print(f"Bob balance: {bob.balance / 10**18} ETH")
    
    # Test transfer
    print("\nTesting transfer...")
    tx_hash = blockchain.transfer(
        alice.address, bob.address, 1 * 10**18  # 1 ETH
    )
    print(f"Transfer transaction hash: {tx_hash}")
    
    # Start mining
    print("\nStarting mining...")
    blockchain.start_mining()
    time.sleep(3)  # Wait for mining
    blockchain.stop_mining()
    
    # Check results
    print(f"\nAlice balance: {blockchain.get_balance(alice.address) / 10**18:.4f} ETH")
    print(f"Bob balance: {blockchain.get_balance(bob.address) / 10**18:.4f} ETH")
    
    # Network information
    network_info = blockchain.get_network_info()
    print(f"\nNetwork information:")
    for key, value in network_info.items():
        print(f"  {key}: {value}")
    
    print("\nBlockchain simulation environment test completed!")