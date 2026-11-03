"""Entities and roles for the quantization proof (port of defense-and-distribution/entity_roles.py).

  Distributor (trusted)  holds FP weights + signing keypair; publishes FP model + Sign(sk_D, root_FP)
  Quantizer   (untrusted) holds FP weights + spec S; publishes INT model + proof
  Verifier    (weak)      holds S, (root_FP, sigma), pk_D, INT model, proof; never the FP model

Base classes define the interface; phase2.py subclasses them with the real Merkle/Ed25519/Fiat-Shamir
implementations. The only key in the system is the distributor's signing keypair.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from spec import QuantSpec


@dataclass(frozen=True)
class FPModel:
    weights: Dict[str, object]
    layer_order: Tuple[str, ...]


@dataclass(frozen=True)
class Commitment:
    root: bytes


@dataclass(frozen=True)
class DistributorPublicKey:
    key: bytes


@dataclass(frozen=True)
class SignedCommitment:
    commitment: Commitment
    signature: bytes


@dataclass(frozen=True)
class IntModel:
    layers: Dict[str, Tuple[object, object]]       # {layer: (scales [out], q [out,in])}
    spec: QuantSpec


@dataclass(frozen=True)
class Proof:
    payload: bytes = b""


class Distributor:
    HOLDS = ("fp_model", "signing_keypair")

    def __init__(self, fp_model: FPModel):
        self.fp_model = fp_model
        self._sk: Optional[bytes] = None
        self.public_key: Optional[DistributorPublicKey] = None

    def generate_keypair(self) -> DistributorPublicKey:
        raise NotImplementedError

    def commit(self) -> Commitment:
        raise NotImplementedError

    def sign(self, commitment: Commitment) -> SignedCommitment:
        raise NotImplementedError

    def publish(self) -> Tuple[FPModel, SignedCommitment, DistributorPublicKey]:
        if self.public_key is None:
            self.public_key = self.generate_keypair()
        return self.fp_model, self.sign(self.commit()), self.public_key


class Quantizer:
    HOLDS = ("fp_model", "spec")

    def __init__(self, fp_model: FPModel, spec: QuantSpec):
        self.fp_model = fp_model
        self.spec = spec

    def quantize(self) -> IntModel:
        from verify import honest_quantize
        layers = {name: honest_quantize(self.fp_model.weights[name], self.spec.qmax)
                  for name in self.fp_model.layer_order}
        return IntModel(layers=layers, spec=self.spec)

    def prove(self, int_model: IntModel, signed: SignedCommitment) -> Proof:
        raise NotImplementedError


class Verifier:
    HOLDS = ("spec", "signed_commitment", "distributor_public_key", "int_model", "proof")

    def __init__(self, spec: QuantSpec, signed: SignedCommitment, distributor_pk: DistributorPublicKey):
        self.spec = spec
        self.signed = signed
        self.distributor_pk = distributor_pk

    def check_signature(self) -> bool:
        raise NotImplementedError

    def check_proof(self, int_model: IntModel, proof: Proof) -> bool:
        raise NotImplementedError


ENTITIES: Sequence[type] = (Distributor, Quantizer, Verifier)
