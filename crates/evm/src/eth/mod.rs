//! Ethereum EVM implementation.

use crate::{env::EvmEnv, evm::EvmFactory, precompiles::PrecompilesMap, Database, Evm};
use alloy_primitives::{Address, Bytes};
use core::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};
use revm::{
    context::{BlockEnv, CfgEnv, Evm as RevmEvm, TxEnv},
    context_interface::result::{EVMError, HaltReason, ResultAndState},
    handler::{instructions::EthInstructions, EthFrame, EthPrecompiles, PrecompileProvider},
    inspector::NoOpInspector,
    interpreter::{interpreter::EthInterpreter, InterpreterResult},
    precompile::{PrecompileSpecId, Precompiles},
    primitives::hardfork::SpecId,
    Context, ExecuteEvm, InspectEvm, Inspector, MainBuilder, MainContext, SystemCallEvm,
};

mod block;
pub use block::*;

pub mod dao_fork;
pub mod eip6110;
pub mod receipt_builder;
pub mod spec;

/// The Ethereum EVM context type.
pub type EthEvmContext<DB> = Context<BlockEnv, TxEnv, CfgEnv, DB>;

/// Newtype wrapper around NoOpInspector for Ethereum EVM.
#[derive(Debug, Default, Clone)]
pub struct EthNoOpInspector(NoOpInspector);

// Safety: NoOpInspector is Send and Sync, so EthNoOpInspector can be too
unsafe impl Send for EthNoOpInspector {}
unsafe impl Sync for EthNoOpInspector {}

/// Ethereum EVM implementation.
///
/// This is a wrapper type around the `revm` ethereum evm with optional [`Inspector`] (tracing)
/// support. [`Inspector`] support is configurable at runtime because it's part of the underlying
/// [`RevmEvm`] type.
#[expect(missing_debug_implementations)]
pub struct EthEvm<DB: Database, I, PRECOMPILE = EthPrecompiles> {
    inner: RevmEvm<
        EthEvmContext<DB>,
        I,
        EthInstructions<EthInterpreter, EthEvmContext<DB>>,
        PRECOMPILE,
        EthFrame,
    >,
    inspect: bool,
}

impl<DB: Database, I, PRECOMPILE> EthEvm<DB, I, PRECOMPILE> {
    /// Creates a new Ethereum EVM instance.
    ///
    /// The `inspect` argument determines whether the configured [`Inspector`] of the given
    /// [`RevmEvm`] should be invoked on [`Evm::transact`].
    pub const fn new(
        evm: RevmEvm<
            EthEvmContext<DB>,
            I,
            EthInstructions<EthInterpreter, EthEvmContext<DB>>,
            PRECOMPILE,
            EthFrame,
        >,
        inspect: bool,
    ) -> Self {
        Self { inner: evm, inspect }
    }

    /// Consumes self and return the inner EVM instance.
    pub fn into_inner(
        self,
    ) -> RevmEvm<
        EthEvmContext<DB>,
        I,
        EthInstructions<EthInterpreter, EthEvmContext<DB>>,
        PRECOMPILE,
        EthFrame,
    > {
        self.inner
    }

    /// Provides a reference to the EVM context.
    pub const fn ctx(&self) -> &EthEvmContext<DB> {
        &self.inner.ctx
    }

    /// Provides a mutable reference to the EVM context.
    pub fn ctx_mut(&mut self) -> &mut EthEvmContext<DB> {
        &mut self.inner.ctx
    }
}

impl<DB: Database, I, PRECOMPILE> Deref for EthEvm<DB, I, PRECOMPILE> {
    type Target = EthEvmContext<DB>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.ctx()
    }
}

impl<DB: Database, I, PRECOMPILE> DerefMut for EthEvm<DB, I, PRECOMPILE> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx_mut()
    }
}

impl<DB, I, PRECOMPILE> Evm for EthEvm<DB, I, PRECOMPILE>
where
    DB: Database,
    I: Inspector<EthEvmContext<DB>>,
    PRECOMPILE: PrecompileProvider<EthEvmContext<DB>, Output = InterpreterResult>,
{
    type DB = DB;
    type Tx = TxEnv;
    type Error = EVMError<DB::Error>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type Precompiles = PRECOMPILE;
    type Inspector = I;

    fn block(&self) -> &BlockEnv {
        &self.block
    }

    fn chain_id(&self) -> u64 {
        self.cfg.chain_id
    }

    fn transact_raw(
        &mut self,
        tx: Self::Tx,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        if self.inspect {
            self.inner.inspect_tx(tx)
        } else {
            self.inner.transact(tx)
        }
    }

    fn transact_system_call(
        &mut self,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        self.inner.transact_system_call_with_caller_finalize(caller, contract, data)
    }

    fn finish(self) -> (Self::DB, EvmEnv<Self::Spec>) {
        let Context { block: block_env, cfg: cfg_env, journaled_state, .. } = self.inner.ctx;

        (journaled_state.database, EvmEnv { block_env, cfg_env })
    }

    fn set_inspector_enabled(&mut self, enabled: bool) {
        self.inspect = enabled;
    }

    fn components(&self) -> (&Self::DB, &Self::Inspector, &Self::Precompiles) {
        (&self.inner.ctx.journaled_state.database, &self.inner.inspector, &self.inner.precompiles)
    }

    fn components_mut(&mut self) -> (&mut Self::DB, &mut Self::Inspector, &mut Self::Precompiles) {
        (
            &mut self.inner.ctx.journaled_state.database,
            &mut self.inner.inspector,
            &mut self.inner.precompiles,
        )
    }
}

/// Factory producing [`EthEvm`].
#[derive(Debug, Default, Clone, Copy)]
#[non_exhaustive]
pub struct EthEvmFactory;

impl EvmFactory for EthEvmFactory {
    type Evm<DB: Database, I: Inspector<Self::Context<DB>>> = EthEvm<DB, I, Self::Precompiles>;
    type Context<DB: Database> = EthEvmContext<DB>;
    type Tx = TxEnv;
    type Error<DBError: core::error::Error + Send + Sync + 'static> = EVMError<DBError>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type Precompiles = PrecompilesMap;
    type DefaultInspector = EthNoOpInspector;

    fn create_evm<DB: Database>(&self, db: DB, input: EvmEnv) -> Self::Evm<DB, Self::DefaultInspector>
    where
        Self::DefaultInspector: Inspector<Self::Context<DB>> + Default + Clone + Send + Sync + 'static,
    {
        let spec_id = input.cfg_env.spec;
        EthEvm {
            inner: Context::mainnet()
                .with_block(input.block_env)
                .with_cfg(input.cfg_env)
                .with_db(db)
                .build_mainnet_with_inspector(Self::DefaultInspector::default())
                .with_precompiles(PrecompilesMap::from_static(Precompiles::new(
                    PrecompileSpecId::from_spec_id(spec_id),
                ))),
            inspect: false,
        }
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        let spec_id = input.cfg_env.spec;
        EthEvm {
            inner: Context::mainnet()
                .with_block(input.block_env)
                .with_cfg(input.cfg_env)
                .with_db(db)
                .build_mainnet_with_inspector(inspector)
                .with_precompiles(PrecompilesMap::from_static(Precompiles::new(
                    PrecompileSpecId::from_spec_id(spec_id),
                ))),
            inspect: true,
        }
    }
}

// Deref implementation for EthNoOpInspector
impl core::ops::Deref for EthNoOpInspector {
    type Target = NoOpInspector;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for EthNoOpInspector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Inspector implementation for EthNoOpInspector - complete NoOp implementation
impl<DB: Database> Inspector<EthEvmContext<DB>> for EthNoOpInspector {
    fn step(&mut self, _interp: &mut revm::interpreter::Interpreter, _context: &mut EthEvmContext<DB>) {
        // NoOp - do nothing
    }

    fn step_end(&mut self, _interp: &mut revm::interpreter::Interpreter, _context: &mut EthEvmContext<DB>) {
        // NoOp - do nothing
    }

    fn call(
        &mut self,
        _context: &mut EthEvmContext<DB>,
        _inputs: &mut revm::interpreter::CallInputs,
    ) -> Option<revm::interpreter::CallOutcome> {
        None // NoOp - no override
    }

    fn call_end(
        &mut self,
        _context: &mut EthEvmContext<DB>,
        _inputs: &revm::interpreter::CallInputs,
        _outcome: &mut revm::interpreter::CallOutcome,
    ) {
        // NoOp - do nothing
    }

    fn create(
        &mut self,
        _context: &mut EthEvmContext<DB>,
        _inputs: &mut revm::interpreter::CreateInputs,
    ) -> Option<revm::interpreter::CreateOutcome> {
        None // NoOp - no override
    }

    fn create_end(
        &mut self,
        _context: &mut EthEvmContext<DB>,
        _inputs: &revm::interpreter::CreateInputs,
        _outcome: &mut revm::interpreter::CreateOutcome,
    ) {
        // NoOp - do nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::address;
    use revm::{database_interface::EmptyDB, primitives::hardfork::SpecId};

    #[test]
    fn test_precompiles_with_correct_spec() {
        // create tests where precompile should be available for later specs but not earlier ones
        let specs_to_test = [
            // MODEXP (0x05) was added in Byzantium, should not exist in Frontier
            (
                address!("0x0000000000000000000000000000000000000005"),
                SpecId::FRONTIER,  // Early spec - should NOT have this precompile
                SpecId::BYZANTIUM, // Later spec - should have this precompile
                "MODEXP",
            ),
            // BLAKE2F (0x09) was added in Istanbul, should not exist in Byzantium
            (
                address!("0x0000000000000000000000000000000000000009"),
                SpecId::BYZANTIUM, // Early spec - should NOT have this precompile
                SpecId::ISTANBUL,  // Later spec - should have this precompile
                "BLAKE2F",
            ),
        ];

        for (precompile_addr, early_spec, later_spec, name) in specs_to_test {
            let mut early_cfg_env = CfgEnv::default();
            early_cfg_env.spec = early_spec;
            early_cfg_env.chain_id = 1;

            let early_env = EvmEnv { block_env: BlockEnv::default(), cfg_env: early_cfg_env };
            let factory = EthEvmFactory;
            let mut early_evm = factory.create_evm(EmptyDB::default(), early_env);

            // precompile should NOT be available in early spec
            assert!(
                early_evm.precompiles_mut().get(&precompile_addr).is_none(),
                "{name} precompile at {precompile_addr:?} should NOT be available for early spec {early_spec:?}"
            );

            let mut later_cfg_env = CfgEnv::default();
            later_cfg_env.spec = later_spec;
            later_cfg_env.chain_id = 1;

            let later_env = EvmEnv { block_env: BlockEnv::default(), cfg_env: later_cfg_env };
            let mut later_evm = factory.create_evm(EmptyDB::default(), later_env);

            // precompile should be available in later spec
            assert!(
                later_evm.precompiles_mut().get(&precompile_addr).is_some(),
                "{name} precompile at {precompile_addr:?} should be available for later spec {later_spec:?}"
            );
        }
    }
}
