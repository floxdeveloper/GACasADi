package de.orat.math.gacasadi.caching;

import de.dhbw.rahmlab.casadi.impl.casadi.Sparsity;
import de.dhbw.rahmlab.casadi.impl.std.StdVectorCasadiInt;
import de.orat.math.gacalc.spi.IGAFunctionSpecializationCache;
import de.orat.math.gacasadi.generic.GaFactory;
import de.orat.math.gacasadi.generic.GaFunction;
import de.orat.math.gacasadi.generic.IGaMvExpr;
import de.orat.math.gacasadi.generic.IGaMvValue;
import de.orat.math.gacasadi.generic.IGaMvVariable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Independent cache for symbolic function specializations.  This deliberately
 * does not share entries or keys with {@link GaFunctionCache}.
 */
public final class GaFunctionSpecializationCache<
    EXPR extends IGaMvExpr<EXPR, VAR, VAL>,
    VAR extends IGaMvVariable<EXPR, VAR, VAL>,
    VAL extends IGaMvValue<EXPR, VAR, VAL>
> implements IGAFunctionSpecializationCache<EXPR, VAR, VAL> {

    private final GaFactory<EXPR, VAR, VAL> factory;
    private final Map<Key, GaFunction<EXPR, VAR, VAL>> entries = new HashMap<>();
    private long sequence = 0;

    public GaFunctionSpecializationCache(GaFactory<EXPR, VAR, VAL> factory) {
        this.factory = factory;
    }

    @Override
    public List<EXPR> executeCached(List<? extends EXPR> arguments,
        String funcName,
        Function<List<VAR>, List<EXPR>> creator) {
        List<EXPR> actualArguments = List.copyOf(arguments);
        Key key = Key.of(actualArguments);
        GaFunction<EXPR, VAR, VAL> function = entries.get(key);
        if (function == null) {
            function = createFunction(actualArguments, funcName, creator);
            entries.put(key, function);
        }
        return function.callExpr(actualArguments); // Returned Expr is already of cached type.
    }

    private GaFunction<EXPR, VAR, VAL> createFunction(List<EXPR> arguments,
        String funcName,
        Function<List<VAR>, List<EXPR>> creator) {
        List<VAR> formalParameters = new ArrayList<>(arguments.size());
        List<VAR> creatorArguments = new ArrayList<>(arguments.size());
        IdentityHashMap<EXPR, VAR> firstFormals = new IdentityHashMap<>();
        for (int i = 0; i < arguments.size(); i++) {
            // Important to include funcName here. Different functions can have arguments with same sparsity.
            VAR formal = arguments.get(i).toVar("cache_" + funcName + "_" + sequence + "_a" + i);
            formalParameters.add(formal);
            // Preserve identiy equality relation in arguments to variables.
            VAR firstFormal = firstFormals.putIfAbsent(arguments.get(i), formal);
            creatorArguments.add(firstFormal == null ? formal : firstFormal);
        }

        List<EXPR> returns = List.copyOf(creator.apply(List.copyOf(creatorArguments)));
        if (returns.isEmpty()) {
            throw new IllegalStateException("A function specialization must return at least one multivector.");
        }
        List<EXPR> simplifiedReturns = returns.stream().parallel()
            .map(result -> result.simplify(formalParameters))
            .toList();
        GaFunction<EXPR, VAR, VAL> function = factory.createFunction(
            "function_specialization_" + funcName + "_" + sequence, formalParameters, simplifiedReturns);
        sequence++;
        return function;
    }

    @Override
    public void clearCache() {
        entries.clear();
    }

    @Override
    public int getCacheSize() {
        return entries.size();
    }

    private static int[] toIntArr(StdVectorCasadiInt intVec) {
        return intVec.stream().mapToInt(Long::intValue).toArray();
    }

    private record Key(List<ArgumentSignature> arguments) {
        static Key of(List<? extends IGaMvExpr<?, ?, ?>> arguments) {
            IdentityHashMap<Object, Integer> firstOccurrences = new IdentityHashMap<>();
            List<ArgumentSignature> signatures = new ArrayList<>(arguments.size());
            for (int i = 0; i < arguments.size(); i++) {
                Object argument = arguments.get(i);
                Integer first = firstOccurrences.putIfAbsent(argument, i);
                Sparsity sparsity = arguments.get(i).getSparsityCasadi();
                signatures.add(new ArgumentSignature(
                    new SparsitySignature((int) sparsity.rows(), (int) sparsity.columns(), toIntArr(sparsity.get_colind()), toIntArr(sparsity.get_row())),
                    first == null ? i : first));
            }
            return new Key(List.copyOf(signatures));
        }
    }

    private record ArgumentSignature(SparsitySignature sparsity, int firstOccurrence) { }

    private record SparsitySignature(int rows, int columns, int[] colind, int[] row) {
        @Override public boolean equals(Object other) {
            return other instanceof SparsitySignature that && rows == that.rows && columns == that.columns
                && Arrays.equals(colind, that.colind) && Arrays.equals(row, that.row);
        }

        @Override public int hashCode() {
            int result = 31 * rows + columns;
            result = 31 * result + Arrays.hashCode(colind);
            return 31 * result + Arrays.hashCode(row);
        }
    }
}
