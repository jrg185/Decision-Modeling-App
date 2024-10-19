import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, milp, OptimizeResult
from scipy.optimize import LinearConstraint, Bounds
import logging

# Set up logging
logging.basicConfig(filename='streamlit_app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_constraint(constraint_expr, variables):
    if '<=' in constraint_expr:
        parts = constraint_expr.split('<=')
        relation = '<='
    elif '>=' in constraint_expr:
        parts = constraint_expr.split('>=')
        relation = '>='
    elif '=' in constraint_expr:
        parts = constraint_expr.split('=')
        relation = '='
    else:
        logger.error(f"Invalid constraint format: {constraint_expr}")
        return None, None, None
    
    if len(parts) != 2:
        logger.error(f"Invalid constraint format: {constraint_expr}")
        return None, None, None

    lhs, rhs = parts[0].strip(), parts[1].strip()
    
    coeffs = [0] * len(variables)
    terms = lhs.split('+')
    for term in terms:
        term = term.strip()
        if '*' in term:
            coeff, var = term.split('*')
            coeff = float(coeff.strip())
        else:
            coeff = 1.0
            var = term
        
        var = var.strip()
        var_index = next((i for i, v in enumerate(variables) if v == var), None)
        if var_index is None:
            logger.error(f"Variable '{var}' not found in defined variables: {variables}")
            return None, None, None
        
        coeffs[var_index] = coeff
    
    try:
        rhs_value = float(rhs)
    except ValueError:
        logger.error(f"Invalid right-hand side value: {rhs}")
        return None, None, None

    return coeffs, rhs_value, relation

def linear_optimize(obj_coeffs, constraints, variables, objective_type):
    c = obj_coeffs if objective_type == "Minimize" else [-coeff for coeff in obj_coeffs]
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for _, constraint in constraints:
        coeffs, rhs, relation = parse_constraint(constraint, variables)
        if relation == "<=":
            A_ub.append(coeffs)
            b_ub.append(rhs)
        elif relation == ">=":
            A_ub.append([-coeff for coeff in coeffs])
            b_ub.append(-rhs)
        else:  # "="
            A_eq.append(coeffs)
            b_eq.append(rhs)
    
    bounds = [(0, None) for _ in range(len(variables))]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex')
    
    return OptimizeResult(
        x=res.x,
        fun=-res.fun if objective_type == "Maximize" else res.fun,
        success=res.success,
        status=res.message,
        message=res.message
    )

def nonlinear_optimize(obj_function, constraints, variables, objective_type):
    def objective(x):
        return eval(obj_function, dict(zip(variables, x)))

    cons = []
    for _, expr in constraints:
        if '<=' in expr:
            lhs, rhs = expr.split('<=')
            cons.append({'type': 'ineq', 'fun': lambda x, l=lhs, r=rhs: eval(r, dict(zip(variables, x))) - eval(l, dict(zip(variables, x)))})
        elif '>=' in expr:
            lhs, rhs = expr.split('>=')
            cons.append({'type': 'ineq', 'fun': lambda x, l=lhs, r=rhs: eval(l, dict(zip(variables, x))) - eval(r, dict(zip(variables, x)))})
        elif '=' in expr:
            lhs, rhs = expr.split('=')
            cons.append({'type': 'eq', 'fun': lambda x, l=lhs, r=rhs: eval(l, dict(zip(variables, x))) - eval(r, dict(zip(variables, x)))})

    x0 = np.ones(len(variables)) / len(variables)  # Initial guess: equal distribution
    bounds = [(0, 1) for _ in variables]  # Assuming all variables are proportions between 0 and 1
    
    res = minimize(objective if objective_type == "Minimize" else lambda x: -objective(x), 
                   x0, method='SLSQP', constraints=cons, bounds=bounds)
    
    return OptimizeResult(
        x=res.x,
        fun=res.fun if objective_type == "Minimize" else -res.fun,
        success=res.success,
        status=res.message,
        message=res.message
    )

def integer_optimize(obj_coeffs, constraints, variables, objective_type):
    logger.debug(f"Starting integer optimization with variables: {variables}")
    logger.debug(f"Constraints: {constraints}")
    
    c = np.array(obj_coeffs if objective_type == "Minimize" else [-coeff for coeff in obj_coeffs])
    
    A = []
    b_lower = []
    b_upper = []
    
    for constraint_name, constraint in constraints:
        logger.debug(f"Processing constraint: {constraint_name}: {constraint}")
        coeffs, rhs_value, relation = parse_constraint(constraint, variables)
        
        if coeffs is None:
            logger.error(f"Failed to parse constraint: {constraint}")
            raise ValueError(f"Invalid constraint format: {constraint}")
        
        A.append(coeffs)
        if relation == "<=":
            b_lower.append(-np.inf)
            b_upper.append(rhs_value)
        elif relation == ">=":
            b_lower.append(rhs_value)
            b_upper.append(np.inf)
    
    A = np.array(A)
    logger.debug(f"Constraint matrix A:\n{A}")
    logger.debug(f"Lower bounds: {b_lower}")
    logger.debug(f"Upper bounds: {b_upper}")
    
    integrality = np.ones(len(variables))  # All variables are integers
    bounds = Bounds(lb=np.zeros(len(variables)), ub=np.inf * np.ones(len(variables)))
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        logger.debug(f"Optimization result: {res}")
        return OptimizeResult(
            x=res.x,
            fun=-res.fun if objective_type == "Maximize" else res.fun,
            success=res.success,
            status=res.status,
            message=res.message
        )
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

def display_results(result, variables, obj_function, objective_type, constraints, model_type):
    st.header("Optimization Results")
    
    # Display optimal solution
    st.subheader("Optimal Solution")
    solution_df = pd.DataFrame({
        'Variable': variables,
        'Optimal Value': result.x.round().astype(int) if model_type == "Integer Programming" else result.x,
    })
    st.dataframe(solution_df)
    
    # Plot the solution using Streamlit's native chart
    st.subheader("Optimal Solution Visualization")
    chart_data = pd.DataFrame({'Variable': variables, 'Value': result.x.round().astype(int) if model_type == "Integer Programming" else result.x})
    st.bar_chart(chart_data.set_index('Variable'))
    
    # Display objective value
    st.subheader("Objective Value")
    st.write(f"The {objective_type.lower()}d value is: {abs(result.fun):.4f}")
    
    # Check constraints
    st.subheader("Constraint Satisfaction")
    for constraint_name, constraint_expr in constraints:
        coeffs, rhs_value, relation = parse_constraint(constraint_expr, variables)
        lhs_value = np.dot(coeffs, result.x)
        
        if relation == '<=':
            satisfied = lhs_value <= rhs_value
        elif relation == '>=':
            satisfied = lhs_value >= rhs_value
        else:  # '='
            satisfied = np.isclose(lhs_value, rhs_value)
        
        st.write(f"{constraint_name}: {'Satisfied' if satisfied else 'Not Satisfied'}")
        st.write(f"  Left-hand side: {lhs_value:.4f}")
        st.write(f"  Relation: {relation}")
        st.write(f"  Right-hand side: {rhs_value:.4f}")
    
    # Optimization status
    st.subheader("Optimization Status")
    st.write(f"Success: {result.success}")
    st.write(f"Message: {result.message}")
    
    if not result.success:
        st.warning("Warning: The optimizer couldn't find a perfect solution. The results might not be optimal or might slightly violate some constraints.")

def main():
    st.title("Decision Optimization Modeling App")

    # Step 1: Determine the type of decision model
    st.header("Step 1: Choose Your Decision Model")
    model_type = st.radio(
        "What type of decision model do you need?",
        ("Linear Programming", "Nonlinear Programming", "Integer Programming")
    )

    # Provide information about each model type
    if model_type == "Linear Programming":
        st.info("""
        Linear Programming is suitable when:
        - Your objective function and constraints are linear (e.g., maximize profit subject to resource constraints).
        - All variables can take on any real value.
        - Relationships between variables are straightforward and proportional.
        
        Example: Optimizing product mix to maximize profit given limited resources.
        """)
    elif model_type == "Nonlinear Programming":
        st.info("""
        Nonlinear Programming is appropriate when:
        - Your objective function or constraints have nonlinear relationships (e.g., quadratic, exponential).
        - You're dealing with more complex systems or behaviors.
        - Variables can take on any real value, but their relationships are not strictly linear.
        
        Example: Optimizing chemical processes where reactions follow nonlinear rates.
        """)
    else:  # Integer Programming
        st.info("""
        Integer Programming is used when:
        - Some or all of your variables must be integers.
        - You're dealing with indivisible units or yes/no decisions.
        - Your problem involves discrete choices.
        
        Example: Determining the optimal number of machines to purchase, where fractional machines don't make sense.
        """)

    # Step 2: Get user inputs
    st.header("Step 2: Model Parameters")
    
    # Number of decision variables
    st.subheader("2.1 Define Your Variables")
    st.write("What are the things you can change in your problem? (e.g., products, resources)")
    decision_vars = st.number_input("How many different variables do you have?", min_value=1, value=2)
    variables = [st.text_input(f"Name for Variable {i+1}", value=f"x{i+1}", key=f"var_name_{i}") for i in range(decision_vars)]
    
    # Objective function
    st.subheader("2.2 Set Up Your Goal")
    objective_type = st.radio("Do you want to maximize or minimize?", ("Maximize", "Minimize"))
    
    if model_type == "Nonlinear Programming":
        st.write("Enter your objective function using Python syntax. Use the variable names you defined above.")
        st.write("Example: 100 * (x1**0.5 + x2**0.5) - 0.05 * (x1**2 + x2**2)")
        obj_function = st.text_input("Objective Function:", value="100 * (x1**0.5 + x2**0.5) - 0.05 * (x1**2 + x2**2)")
    else:
        st.write("For each variable, how much does it contribute to your goal?")
        obj_coeffs = [st.number_input(f"Value per unit of {var}", value=1.0, key=f"obj_{i}") for i, var in enumerate(variables)]
    
    # Display the objective function
    if model_type != "Nonlinear Programming":
        obj_function = " + ".join([f"{coeff} * {var}" for var, coeff in zip(variables, obj_coeffs) if coeff != 0])
        st.write(f"Your goal: {objective_type} Z = {obj_function}")

    # Constraints
    st.subheader("2.3 Add Your Limits")
    st.write("What restrictions or limits do you have? (e.g., budget, time, space)")
    num_constraints = st.number_input("How many limits do you have?", min_value=0, value=1)
    constraints = []
    for i in range(num_constraints):
        st.write(f"Limit {i+1}:")
        constraint_name = st.text_input(f"Name of this limit (e.g., 'Budget', 'Time available')", key=f"cons_name_{i}")
        if model_type == "Nonlinear Programming":
            constraint_expr = st.text_input(f"Enter constraint (e.g., x1 + x2 <= 10)", key=f"cons_expr_{i}")
        else:
            constraint_coeffs = [st.number_input(f"How much of this limit does one unit of {var} use?", value=0.0, key=f"cons_{i}_{j}") for j, var in enumerate(variables)]
            relation = st.selectbox("Type of limit", ["<=", "=", ">="], key=f"rel_{i}")
            rhs = st.number_input("Total amount available for this limit", value=0.0, key=f"rhs_{i}")
            
            # Display the constraint
            constraint_expr = " + ".join([f"{coeff} * {var}" for var, coeff in zip(variables, constraint_coeffs) if coeff != 0])
            constraint_expr += f" {relation} {rhs}"
        
        constraints.append((constraint_name, constraint_expr))
        st.write(f"Limit {i+1} ({constraint_name}): {constraint_expr}")

    # Step 3: Optimization and visualization
    if st.button("Find the Best Solution"):
        try:
            if model_type == "Linear Programming":
                result = linear_optimize(obj_coeffs, constraints, variables, objective_type)
            elif model_type == "Nonlinear Programming":
                result = nonlinear_optimize(obj_function, constraints, variables, objective_type)
            else:  # Integer Programming
                result = integer_optimize(obj_coeffs, constraints, variables, objective_type)
            
            display_results(result, variables, obj_function if model_type == "Nonlinear Programming" else obj_coeffs, objective_type, constraints, model_type)
        except Exception as e:
            st.error(f"An error occurred during optimization: {str(e)}")
            logger.exception("Error during optimization")
            st.write("Debug Information:")
            st.write(f"Objective Type: {objective_type}")
            st.write(f"Objective Function: {obj_function if model_type == 'Nonlinear Programming' else obj_coeffs}")
            st.write(f"Constraints: {constraints}")
            st.write(f"Decision Variables: {len(variables)}")
            st.write(f"Variables: {variables}")
            st.write(f"Model Type: {model_type}")
            
            # Display the log
            st.write("Log Output:")
            with open("streamlit_app.log", "r") as log_file:
                st.code(log_file.read())

if __name__ == "__main__":
    main()
