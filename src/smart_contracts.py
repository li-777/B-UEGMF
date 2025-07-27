"""
Smart contract implementation
Contains smart contracts for access control, NFT management and data permission management
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from blockchain_simulation import SimulatedBlockchain, Event
from config import Config

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access permission levels"""
    UNAUTHORIZED = 0
    READ_ONLY = 1
    FULL_ACCESS = 2
    ADMIN = 3


@dataclass
class NFTMetadata:
    """NFT metadata"""
    token_id: int
    owner: str
    dataset_hash: str
    access_level: AccessLevel
    ipfs_cid: str
    created_at: int
    expires_at: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'token_id': self.token_id,
            'owner': self.owner,
            'dataset_hash': self.dataset_hash,
            'access_level': self.access_level.value,
            'ipfs_cid': self.ipfs_cid,
            'created_at': self.created_at,
            'expires_at': self.expires_at
        }


@dataclass
class MerkleProof:
    """Merkle proof"""
    leaf: str
    proof_hashes: List[str]
    root: str
    
    def verify(self) -> bool:
        """Verify Merkle proof"""
        current_hash = self.leaf
        
        for proof_hash in self.proof_hashes:
            # Determine left/right order based on hash value
            if current_hash < proof_hash:
                combined = current_hash + proof_hash
            else:
                combined = proof_hash + current_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == self.root


class UEAccessControlContract:
    """Unlearnable Example Access Control Smart Contract"""
    
    def __init__(self, blockchain: SimulatedBlockchain, contract_address: str):
        self.blockchain = blockchain
        self.contract_address = contract_address
        
        # Contract state
        self.owner = None
        self.admins: set = set()
        self.merkle_root = "0x0"
        self.nft_counter = 0
        self.nft_metadata: Dict[int, NFTMetadata] = {}
        self.user_tokens: Dict[str, List[int]] = {}  # Mapping from user address to token list
        self.dataset_registry: Dict[str, Dict[str, Any]] = {}  # Dataset registry
        self.access_logs: List[Dict[str, Any]] = []
        
        # Multi-signature configuration
        self.multisig_threshold = 2
        self.multisig_proposals: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Access control contract initialized: {contract_address}")
    
    def initialize(self, owner: str, initial_admins: List[str] = None) -> Dict[str, Any]:
        """Initialize contract"""
        if self.owner is not None:
            raise ValueError("Contract already initialized")
        
        self.owner = owner
        
        if initial_admins:
            self.admins.update(initial_admins)
        
        # Emit initialization event
        self.blockchain.emit_event(
            self.contract_address,
            "ContractInitialized",
            {
                'owner': owner,
                'admins': list(self.admins),
                'timestamp': int(time.time())
            }
        )
        
        logger.info(f"Contract initialized, owner: {owner}")
        return {'status': 'success', 'owner': owner}
    
    def update_merkle_root(self, new_root: str, proposer: str, 
                          signatures: List[str] = None) -> Dict[str, Any]:
        """Update Merkle root (requires multi-signature)"""
        if proposer not in self.admins and proposer != self.owner:
            raise ValueError("Only admins can propose Merkle root updates")
        
        proposal_id = hashlib.sha256(f"{new_root}{proposer}{time.time()}".encode()).hexdigest()
        
        # Create multi-signature proposal
        proposal = {
            'proposal_id': proposal_id,
            'type': 'update_merkle_root',
            'new_root': new_root,
            'proposer': proposer,
            'signatures': signatures or [proposer],
            'created_at': int(time.time()),
            'executed': False
        }
        
        self.multisig_proposals[proposal_id] = proposal
        
        # Check if threshold reached
        if len(proposal['signatures']) >= self.multisig_threshold:
            self.merkle_root = new_root
            proposal['executed'] = True
            
            # Emit event
            self.blockchain.emit_event(
                self.contract_address,
                "MerkleRootUpdated",
                {
                    'old_root': self.merkle_root,
                    'new_root': new_root,
                    'proposal_id': proposal_id,
                    'timestamp': int(time.time())
                }
            )
            
            logger.info(f"Merkle root updated: {new_root}")
            return {'status': 'executed', 'new_root': new_root}
        else:
            logger.info(f"Merkle root update proposal created: {proposal_id}")
            return {'status': 'proposed', 'proposal_id': proposal_id}
    
    def sign_proposal(self, proposal_id: str, signer: str) -> Dict[str, Any]:
        """Sign multi-signature proposal"""
        if proposal_id not in self.multisig_proposals:
            raise ValueError("Proposal does not exist")
        
        proposal = self.multisig_proposals[proposal_id]
        
        if proposal['executed']:
            raise ValueError("Proposal already executed")
        
        if signer not in self.admins and signer != self.owner:
            raise ValueError("Only admins can sign proposals")
        
        if signer not in proposal['signatures']:
            proposal['signatures'].append(signer)
        
        # Check if execution threshold reached
        if len(proposal['signatures']) >= self.multisig_threshold:
            if proposal['type'] == 'update_merkle_root':
                self.merkle_root = proposal['new_root']
                proposal['executed'] = True
                
                self.blockchain.emit_event(
                    self.contract_address,
                    "MerkleRootUpdated",
                    {
                        'new_root': proposal['new_root'],
                        'proposal_id': proposal_id,
                        'timestamp': int(time.time())
                    }
                )
        
        return {'status': 'signed', 'signatures_count': len(proposal['signatures'])}
    
    def grant_access(self, user: str, merkle_proof: MerkleProof, 
                    dataset_hash: str, access_level: AccessLevel,
                    ipfs_cid: str, expiry_time: Optional[int] = None) -> Dict[str, Any]:
        """Grant access (mint NFT)"""
        # Verify Merkle proof
        if not merkle_proof.verify() or merkle_proof.root != self.merkle_root:
            raise ValueError("Merkle proof verification failed")
        
        # Check if user is in whitelist
        user_hash = hashlib.sha256(user.encode()).hexdigest()
        if merkle_proof.leaf != user_hash:
            raise ValueError("User not in whitelist")
        
        # Create NFT
        self.nft_counter += 1
        token_id = self.nft_counter
        
        nft_metadata = NFTMetadata(
            token_id=token_id,
            owner=user,
            dataset_hash=dataset_hash,
            access_level=access_level,
            ipfs_cid=ipfs_cid,
            created_at=int(time.time()),
            expires_at=expiry_time
        )
        
        self.nft_metadata[token_id] = nft_metadata
        
        # Update user token list
        if user not in self.user_tokens:
            self.user_tokens[user] = []
        self.user_tokens[user].append(token_id)
        
        # Record access log
        access_log = {
            'user': user,
            'token_id': token_id,
            'dataset_hash': dataset_hash,
            'access_level': access_level.value,
            'action': 'grant_access',
            'timestamp': int(time.time()),
            'ipfs_cid': ipfs_cid
        }
        self.access_logs.append(access_log)
        
        # Emit event
        self.blockchain.emit_event(
            self.contract_address,
            "AccessGranted",
            {
                'user': user,
                'token_id': token_id,
                'dataset_hash': dataset_hash,
                'access_level': access_level.value,
                'timestamp': int(time.time())
            }
        )
        
        logger.info(f"Access granted to user {user}, token ID: {token_id}")
        return {
            'status': 'success',
            'token_id': token_id,
            'nft_metadata': nft_metadata.to_dict()
        }
    
    def revoke_access(self, token_id: int, revoker: str) -> Dict[str, Any]:
        """Revoke access (burn NFT)"""
        if token_id not in self.nft_metadata:
            raise ValueError("Token does not exist")
        
        nft = self.nft_metadata[token_id]
        
        # Only owner, admin or token owner can revoke
        if revoker not in [self.owner, nft.owner] and revoker not in self.admins:
            raise ValueError("No permission to revoke this token")
        
        # Remove from user token list
        if nft.owner in self.user_tokens:
            self.user_tokens[nft.owner].remove(token_id)
        
        # Record access log
        access_log = {
            'user': nft.owner,
            'token_id': token_id,
            'dataset_hash': nft.dataset_hash,
            'action': 'revoke_access',
            'revoker': revoker,
            'timestamp': int(time.time())
        }
        self.access_logs.append(access_log)
        
        # Delete NFT metadata
        del self.nft_metadata[token_id]
        
        # Emit event
        self.blockchain.emit_event(
            self.contract_address,
            "AccessRevoked",
            {
                'user': nft.owner,
                'token_id': token_id,
                'revoker': revoker,
                'timestamp': int(time.time())
            }
        )
        
        logger.info(f"Token {token_id} access revoked by {revoker}")
        return {'status': 'success', 'revoked_token_id': token_id}
    
    def check_access(self, user: str, dataset_hash: str) -> Dict[str, Any]:
        """Check user access permissions"""
        user_token_ids = self.user_tokens.get(user, [])
        valid_tokens = []
        
        current_time = int(time.time())
        
        for token_id in user_token_ids:
            if token_id in self.nft_metadata:
                nft = self.nft_metadata[token_id]
                
                # Check dataset match
                if nft.dataset_hash == dataset_hash:
                    # Check expiration
                    if nft.expires_at is None or nft.expires_at > current_time:
                        valid_tokens.append({
                            'token_id': token_id,
                            'access_level': nft.access_level.value,
                            'expires_at': nft.expires_at
                        })
        
        # Record access check log
        access_log = {
            'user': user,
            'dataset_hash': dataset_hash,
            'action': 'check_access',
            'valid_tokens': len(valid_tokens),
            'timestamp': current_time
        }
        self.access_logs.append(access_log)
        
        has_access = len(valid_tokens) > 0
        max_access_level = max([token['access_level'] for token in valid_tokens]) if valid_tokens else 0
        
        return {
            'has_access': has_access,
            'access_level': max_access_level,
            'valid_tokens': valid_tokens,
            'user': user,
            'dataset_hash': dataset_hash
        }
    
    def register_dataset(self, dataset_hash: str, metadata: Dict[str, Any], 
                        registerer: str) -> Dict[str, Any]:
        """Register dataset"""
        if registerer not in self.admins and registerer != self.owner:
            raise ValueError("Only admins can register datasets")
        
        if dataset_hash in self.dataset_registry:
            raise ValueError("Dataset already registered")
        
        dataset_info = {
            'hash': dataset_hash,
            'metadata': metadata,
            'registerer': registerer,
            'registered_at': int(time.time()),
            'access_count': 0
        }
        
        self.dataset_registry[dataset_hash] = dataset_info
        
        # Emit event
        self.blockchain.emit_event(
            self.contract_address,
            "DatasetRegistered",
            {
                'dataset_hash': dataset_hash,
                'registerer': registerer,
                'timestamp': int(time.time())
            }
        )
        
        logger.info(f"Dataset registered: {dataset_hash}")
        return {'status': 'success', 'dataset_hash': dataset_hash}
    
    def get_dataset_info(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Get dataset information"""
        return self.dataset_registry.get(dataset_hash)
    
    def get_user_tokens(self, user: str) -> List[Dict[str, Any]]:
        """Get all tokens for a user"""
        user_token_ids = self.user_tokens.get(user, [])
        tokens = []
        
        for token_id in user_token_ids:
            if token_id in self.nft_metadata:
                tokens.append(self.nft_metadata[token_id].to_dict())
        
        return tokens
    
    def get_access_logs(self, user: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access logs"""
        logs = self.access_logs
        
        if user:
            logs = [log for log in logs if log.get('user') == user]
        
        # Return most recent logs
        return logs[-limit:]
    
    def get_contract_stats(self) -> Dict[str, Any]:
        """Get contract statistics"""
        current_time = int(time.time())
        active_tokens = 0
        expired_tokens = 0
        
        for nft in self.nft_metadata.values():
            if nft.expires_at is None or nft.expires_at > current_time:
                active_tokens += 1
            else:
                expired_tokens += 1
        
        return {
            'total_tokens_minted': self.nft_counter,
            'active_tokens': active_tokens,
            'expired_tokens': expired_tokens,
            'total_users': len(self.user_tokens),
            'registered_datasets': len(self.dataset_registry),
            'total_access_logs': len(self.access_logs),
            'merkle_root': self.merkle_root,
            'admins_count': len(self.admins),
            'owner': self.owner
        }


class MerkleTreeManager:
    """Merkle tree manager"""
    
    def __init__(self):
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        self.root: str = "0x0"
    
    def add_leaf(self, data: str) -> None:
        """Add leaf node"""
        leaf_hash = hashlib.sha256(data.encode()).hexdigest()
        self.leaves.append(leaf_hash)
    
    def build_tree(self) -> str:
        """Build Merkle tree"""
        if not self.leaves:
            self.root = "0x0"
            return self.root
        
        self.tree = [self.leaves.copy()]
        
        while len(self.tree[-1]) > 1:
            current_level = self.tree[-1]
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Ensure consistent ordering
                if left <= right:
                    combined = left + right
                else:
                    combined = right + left
                
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)
            
            self.tree.append(next_level)
        
        self.root = self.tree[-1][0]
        return self.root
    
    def generate_proof(self, leaf: str) -> Optional[MerkleProof]:
        """Generate Merkle proof"""
        if not self.tree:
            return None
        
        leaf_hash = hashlib.sha256(leaf.encode()).hexdigest()
        
        if leaf_hash not in self.leaves:
            return None
        
        proof_hashes = []
        index = self.leaves.index(leaf_hash)
        
        for level in self.tree[:-1]:  # All levels except root
            if index % 2 == 0:  # Left node
                sibling_index = index + 1
            else:  # Right node
                sibling_index = index - 1
            
            if sibling_index < len(level):
                proof_hashes.append(level[sibling_index])
            else:
                proof_hashes.append(level[index])  # Self as sibling node
            
            index //= 2
        
        return MerkleProof(
            leaf=leaf_hash,
            proof_hashes=proof_hashes,
            root=self.root
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify Merkle proof"""
        return proof.verify() and proof.root == self.root


def deploy_access_control_contract(blockchain: SimulatedBlockchain, 
                                  deployer: str) -> Tuple[str, UEAccessControlContract]:
    """Deploy access control contract"""
    # Simulated contract code
    contract_code = f"""
    pragma solidity ^{Config.Blockchain.SOLIDITY_VERSION};
    
    contract UEAccessControl {{
        address public owner;
        bytes32 public merkleRoot;
        uint256 public tokenCounter;
        
        mapping(uint256 => address) public tokenOwner;
        mapping(address => uint256[]) public userTokens;
        
        event AccessGranted(address indexed user, uint256 indexed tokenId);
        event AccessRevoked(address indexed user, uint256 indexed tokenId);
        event MerkleRootUpdated(bytes32 newRoot);
        
        constructor() {{
            owner = msg.sender;
        }}
        
        function grantAccess(address user, bytes32[] calldata proof) external;
        function revokeAccess(uint256 tokenId) external;
        function checkAccess(address user) external view returns (bool);
    }}
    """
    
    # Deploy contract
    contract_address = blockchain.deploy_contract(contract_code, deployer)
    
    # Create contract instance
    contract_instance = UEAccessControlContract(blockchain, contract_address)
    
    logger.info(f"Access control contract deployed: {contract_address}")
    return contract_address, contract_instance


if __name__ == "__main__":
    # Test smart contract
    logging.basicConfig(level=logging.INFO)
    
    from blockchain_simulation import SimulatedBlockchain
    
    print("Testing smart contract...")
    
    # Create blockchain
    blockchain = SimulatedBlockchain()
    accounts = list(blockchain.accounts.values())
    
    # Deploy contract
    deployer = accounts[0].address
    contract_address, contract = deploy_access_control_contract(blockchain, deployer)
    
    # Initialize contract
    contract.initialize(deployer, [accounts[1].address])
    
    # Create Merkle tree
    merkle_manager = MerkleTreeManager()
    
    # Add whitelisted users
    whitelist_users = [accounts[i].address for i in range(5)]
    for user in whitelist_users:
        merkle_manager.add_leaf(user)
    
    # Build Merkle tree
    root = merkle_manager.build_tree()
    print(f"Merkle root: {root}")
    
    # Update Merkle root in contract
    contract.update_merkle_root(root, deployer)
    
    # Register dataset
    dataset_hash = hashlib.sha256("test_dataset".encode()).hexdigest()
    contract.register_dataset(
        dataset_hash,
        {"name": "Test Dataset", "size": 1000},
        deployer
    )
    
    # Grant access to user
    user = whitelist_users[2]
    proof = merkle_manager.generate_proof(user)
    
    if proof:
        result = contract.grant_access(
            user,
            proof,
            dataset_hash,
            AccessLevel.FULL_ACCESS,
            "QmTestCID123"
        )
        print(f"Access grant result: {result}")
        
        # Check access
        access_check = contract.check_access(user, dataset_hash)
        print(f"Access check: {access_check}")
        
        # Get user tokens
        user_tokens = contract.get_user_tokens(user)
        print(f"User tokens: {user_tokens}")
    
    # Get contract stats
    stats = contract.get_contract_stats()
    print(f"Contract stats: {stats}")
    
    print("Smart contract testing completed!")