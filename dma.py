import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize, milp, OptimizeResult
from scipy.optimize import milp, LinearConstraint, Bounds
import logging

# Set up logging
logging.basicConfig(filename='streamlit_app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    variables = [st.text_input(f"Name for Variable {i+1}", value=f"Var{i+1}", key=f"var_name_{i}") for i in range(decision_vars)]
    
    # Objective function
    st.subheader("2.2 Set Up Your Goal")
    objective_type = st.radio("Do you want to maximize or minimize?", ("Maximize", "Minimize"))
    st.write("For each variable, how much does it contribute to your goal?")
    obj_coeffs = [st.number_input(f"Value per unit of {var}", value=1.0, key=f"obj_{i}") for i, var in enumerate(variables)]
    
    # Display the objective function
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
        constraint_coeffs = [st.number_input(f"How much of this limit does one unit of {var} use?", value=0.0, key=f"cons_{i}_{j}") for j, var in enumerate(variables)]
        relation = st.selectbox("Type of limit", ["<=", "=", ">="], key=f"rel_{i}")
        rhs = st.number_input("Total amount available for this limit", value=0.0, key=f"rhs_{i}")
        
        # Display the constraint
        constraint = " + ".join([f"{coeff} * {var}" for var, coeff in zip(variables, constraint_coeffs) if coeff != 0])
        constraint += f" {relation} {rhs}"
        constraints.append(constraint)
        st.write(f"Limit {i+1} ({constraint_name}): {constraint}")

    # Step 3: Optimization and visualization
    if st.button("Find the Best Solution"):
        try:
            result = optimize_model(objective_type, obj_coeffs, constraints, decision_vars, variables, model_type)
            display_results(result, variables, obj_coeffs, objective_type, constraints, model_type)
        except Exception as e:
            st.error(f"An error occurred during optimization: {str(e)}")
            logger.exception("Error during optimization")
            st.write("Debug Information:")
            st.write(f"Objective Type: {objective_type}")
            st.write(f"Objective Coefficients: {obj_coeffs}")
            st.write(f"Constraints: {constraints}")
            st.write(f"Decision Variables: {decision_vars}")
            st.write(f"Variables: {variables}")
            st.write(f"Model Type: {model_type}")
            
            # Display the log
            st.write("Log Output:")
            with open("streamlit_app.log", "r") as log_file:
                st.code(log_file.read())

def parse_coefficients(constraint, variables):
    if '<=' in constraint:
        lhs, rhs = constraint.split('<=')
        relation = '<='
    elif '>=' in constraint:
        lhs, rhs = constraint.split('>=')
        relation = '>='
    elif '=' in constraint:
        lhs, rhs = constraint.split('=')
        relation = '='
    else:
        st.error(f"Invalid constraint format: {constraint}")
        return None, None, None

    lhs = lhs.strip()
    rhs = rhs.strip()
    
    coeffs = []
    for var in variables:
        if var in lhs:
            parts = lhs.split(var)
            coeff_str = parts[0].strip().replace('+', '')
            if coeff_str == '':
                coeffs.append(1.0)
            elif coeff_str == '-':
                coeffs.append(-1.0)
            else:
                try:
                    coeff_str = coeff_str.replace('*', '').strip()
                    coeffs.append(float(coeff_str))
                except ValueError:
                    st.error(f"Invalid coefficient in constraint: {coeff_str}")
                    return None, None, None
            lhs = ''.join(parts[1:]).strip()
        else:
            coeffs.append(0.0)
    
    try:
        rhs_value = float(rhs)
    except ValueError:
        st.error(f"Invalid right-hand side value in constraint: {rhs}")
        return None, None, None
    
    return coeffs, rhs_value, relation

def optimize_model(objective_type, obj_coeffs, constraints, decision_vars, variables, model_type):
    logger.debug(f"Starting optimization with {decision_vars} variables and {len(constraints)} constraints")
    
    if model_type == "Integer Programming":
        # Use milp for integer programming
        c = obj_coeffs if objective_type == "Minimize" else [-coeff for coeff in obj_coeffs]
        
        constraint_matrix = []
        constraint_lb = []
        constraint_ub = []
        
        for constraint in constraints:
            coeffs, rhs_value, relation = parse_coefficients(constraint, variables)
            if coeffs is None:
                raise ValueError(f"Invalid constraint: {constraint}")
            
            constraint_matrix.append(coeffs)
            if relation == '<=':
                constraint_lb.append(-np.inf)
                constraint_ub.append(rhs_value)
            elif relation == '>=':
                constraint_lb.append(rhs_value)
                constraint_ub.append(np.inf)
            else:  # '='
                constraint_lb.append(rhs_value)
                constraint_ub.append(rhs_value)
        
        integrality = np.ones(decision_vars)  # All variables are integer
        bounds = Bounds(lb=np.zeros(decision_vars), ub=np.inf * np.ones(decision_vars))
        
        constraints = LinearConstraint(constraint_matrix, constraint_lb, constraint_ub)
        
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        # Convert the result to a format similar to minimize() output
        return OptimizeResult(x=result.x, fun=result.fun if objective_type == "Minimize" else -result.fun,
                              success=result.success, status=result.status,
                              message=result.message)
    else:
        # Use minimize for linear and nonlinear programming
        def objective(x):
            return np.dot(obj_coeffs, x) * (-1 if objective_type == "Maximize" else 1)

        cons = []
        for constraint in constraints:
            coeffs, rhs_value, relation = parse_coefficients(constraint, variables)
            if coeffs is None:
                raise ValueError(f"Invalid constraint: {constraint}")
            
            if relation == '<=':
                cons.append({'type': 'ineq', 'fun': lambda x, coef=coeffs, r=rhs_value: r - np.dot(coef, x)})
            elif relation == '>=':
                cons.append({'type': 'ineq', 'fun': lambda x, coef=coeffs, r=rhs_value: np.dot(coef, x) - r})
            else:  # '='
                cons.append({'type': 'eq', 'fun': lambda x, coef=coeffs, r=rhs_value: np.dot(coef, x) - r})

        bounds = [(0, None) for _ in range(decision_vars)]
        x0 = np.zeros(decision_vars)
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
            logger.debug(f"Optimization result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise RuntimeError(f"Optimization failed: {str(e)}")

def sensitivity_analysis(result, obj_coeffs, constraints, variables):
    sensitivity = {}
    for i, var in enumerate(variables):
        # Objective coefficient sensitivity
        delta = 0.01 * abs(obj_coeffs[i])
        obj_coeffs_plus = obj_coeffs.copy()
        obj_coeffs_plus[i] += delta
        result_plus = optimize_model("Maximize", obj_coeffs_plus, constraints, len(variables), variables, "Linear Programming")
        sensitivity[f"{var}_obj_coeff"] = (result_plus.fun - result.fun) / delta

        # Constraint sensitivity (for the first constraint only, as an example)
        if constraints:
            coeffs, rhs_value, relation = parse_coefficients(constraints[0], variables)
            if coeffs is None:
                continue
            
            constraints_plus = constraints.copy()
            new_rhs = str(float(rhs_value) + delta)
            constraints_plus[0] = constraints_plus[0].replace(str(rhs_value), new_rhs)
            result_plus = optimize_model("Maximize", obj_coeffs, constraints_plus, len(variables), variables, "Linear Programming")
            sensitivity[f"{var}_constraint"] = (result_plus.fun - result.fun) / delta

    return sensitivity

def generate_answer_report(result, variables, obj_coeffs, objective_type, constraints, model_type):
    report = []
    
    # Objective cell
    obj_value = np.dot(obj_coeffs, result.x)
    report.append({
        'Cell': 'Objective',
        'Name': f'{objective_type} of Objective',
        'Value': obj_value
    })
    
    # Variable cells
    for var, value in zip(variables, result.x):
        report.append({
            'Cell': 'Variable',
            'Name': var,
            'Value': value if model_type != "Integer Programming" else int(round(value))
        })
    
    # Constraints
    for i, constraint in enumerate(constraints):
        coeffs, rhs_value, relation = parse_coefficients(constraint, variables)
        if coeffs is None:
            continue
        
        lhs_value = np.dot(coeffs, result.x)
        status = 'Binding' if np.isclose(lhs_value, rhs_value) else 'Not Binding'
        slack = rhs_value - lhs_value if relation == '<=' else lhs_value - rhs_value if relation == '>=' else 0
        
        report.append({
            'Cell': 'Constraint',
            'Name': f'Constraint {i+1}',
            'Value': lhs_value,
            'Status': status,
            'Slack': slack
        })
    
    return pd.DataFrame(report)

def display_results(result, variables, obj_coeffs, objective_type, constraints, model_type):
    st.header("Optimization Results")
    
    # Round results to integers for Integer Programming
    if model_type == "Integer Programming":
        result.x = np.round(result.x).astype(int)
    
    # Display optimal solution
    st.subheader("Optimal Solution")
    solution_df = pd.DataFrame({
        'Variable': variables,
        'Optimal Value': result.x,
        'Contribution': result.x * obj_coeffs
    })
    st.dataframe(solution_df)
    
    # Plot the solution using Streamlit's native chart
    st.subheader("Optimal Solution Visualization")
    chart_data = pd.DataFrame({'Variable': variables, 'Value': result.x})
    st.bar_chart(chart_data.set_index('Variable'))
    
    # Display objective value
    obj_value = np.dot(obj_coeffs, result.x)
    st.subheader("Objective Value")
    st.write(f"The {objective_type.lower()}d value is: {abs(obj_value):.4f}")
    
    # Check constraints
    st.subheader("Constraint Satisfaction")
    for i, constraint in enumerate(constraints):
        coeffs, rhs_value, relation = parse_coefficients(constraint, variables)
        if coeffs is None:
            continue
        
        lhs_value = np.dot(coeffs, result.x)
        satisfied = (relation == '<=' and lhs_value <= rhs_value) or \
                    (relation == '>=' and lhs_value >= rhs_value) or \
                    (relation == '=' and np.isclose(lhs_value, rhs_value))
        
        st.write(f"Constraint {i+1}: {'Satisfied' if satisfied else 'Not Satisfied'}")
        st.write(f"  Left-hand side: {lhs_value:.4f}")
        st.write(f"  Relation: {relation}")
        st.write(f"  Right-hand side: {rhs_value:.4f}")
    
    # Answer Report
    st.subheader("Answer Report")
    answer_report = generate_answer_report(result, variables, obj_coeffs, objective_type, constraints, model_type)
    st.dataframe(answer_report)
    
    # Sensitivity analysis
    if model_type != "Integer Programming":
        st.subheader("Sensitivity Analysis")
        sensitivity = sensitivity_analysis(result, obj_coeffs, constraints, variables)
        sensitivity_df = pd.DataFrame(sensitivity.items(), columns=['Parameter', 'Sensitivity'])
        st.dataframe(sensitivity_df)
    else:
        st.write("Sensitivity analysis is not applicable for Integer Programming.")
    
    # Optimization status
    st.subheader("Optimization Status")
    st.write(f"Success: {result.success}")
    st.write(f"Message: {result.message}")
    
    if not result.success:
        st.warning("Warning: The optimizer couldn't find a perfect solution. The results might not be optimal or might slightly violate some constraints.")

if __name__ == "__main__":
    main()