use clap::Parser;
use htn_playing::{Method, Operator, PlanningErr, Problem, StateName, StateValue, Task, WorldState};
use std::cell::RefCell;
use std::error::Error;
use std::io::{Write};
use std::ops::Deref;
use std::process::exit;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    domain_file: Option<std::path::PathBuf>,
    #[arg(long)]
    problem_file: Option<std::path::PathBuf>,
    #[arg(long)]
    trace_file: Option<std::path::PathBuf>,
    #[arg(long, default_value = "-")]
    plan_file: Option<std::path::PathBuf>,
    #[arg(long)]
    final_state_file: Option<std::path::PathBuf>,
    #[arg(long, default_value = "false", requires("trace_file"))]
    stepped: bool,
}

struct Tracer {
    output: Box<dyn std::io::Write>,
    stepped: bool,
}

impl Tracer {
    fn trace(&mut self, output: String) {
        writeln!(self.output, "{}", output)
            .expect("writing to trace output");

        if self.stepped {
            print!("stepping, press any key to continue...");
            std::io::stdout().flush().expect("flushing stdout");
            let mut buf = String::new();
            std::io::stdin().read_line(&mut buf)
                .expect("waiting for keypress");
        }
    }
}

impl htn_playing::Tracer for Tracer {
    fn plan_into(&mut self, tasks: &mut Vec<Task>, problem: &Problem) {
        self.trace(format!("plan into: plan so far = {:#?} tasks = {:?}", tasks, problem.tasks));
    }

    fn pop_task(&mut self, task: &Task) {
        self.trace(format!("pop task: task = {:?}", task));
    }

    fn task_planned(&mut self, task: &Task, state: &Rc<RefCell<WorldState>>) {
        self.trace(format!("task planned: task = {:?} state = {:?}", task, state))
    }

    fn has_operator(&mut self, operator: &Operator) {
        self.trace(format!("has operator: operator = {:?}", operator));
    }

    fn has_method(&mut self, method: &Method, state: &Rc<RefCell<WorldState>>) {
        self.trace(format!("has method: method = {:#?} state = {:?}", method, state));
    }

    fn has_applicable_method(&mut self, method: &Method) {
        self.trace(format!("has applicable method: method = {:#?}", method));
    }

    fn expr_evaluated(&mut self, expr: &htn_playing::Expr, result: &Result<StateValue, PlanningErr>) {
        self.trace(format!("evaluated expr {:?} resulting in {:?}", expr, result));
    }

    fn unset_state(&mut self, name: &StateName) {
        self.trace(format!("unset state {:?}", name));
    }

    fn set_state(&mut self, name: &StateName, value: &StateValue) {
        self.trace(format!("set state {:?} to {:?}", name, value));
    }
}

fn main() {
    if let Err(e) = main_err() {
        eprintln!("error: {}", e);
    }
}

fn main_err() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let inputs_using_stdin = || -> Vec<String> {
        let inputs = [
            ("--domain-file", &args.domain_file),
            ("--problem-file", &args.problem_file),
        ];
        inputs.into_iter()
            .filter_map(|(name, path)| {
                match path {
                    Some(p) => Some((name, p)),
                    None => None,
                }
            })
            .filter(|(_, path)| path.as_os_str() == "-")
            .map(|(name, _)| name.into())
            .collect::<Vec<String>>()
    }();

    if inputs_using_stdin.len() > 1 {
        eprintln!("only one input can read from stdin, got: [{}]", inputs_using_stdin.join(", "));
        exit(-1);
    }

    let domain_src = args.domain_file
        .map(|p| {
            if p.as_os_str() == "-" {
                std::io::read_to_string(std::io::stdin())
                    .expect("unable to read domain from stdin")
            } else {
                std::fs::read_to_string(p)
                    .expect("unable to read domain file")
            }
        })
        .unwrap_or_else(|| htn_playing::EXAMPLE_DOMAIN.into());

    let problem_src = args.problem_file
        .map(|p| {
            if p.as_os_str() == "-" {
                std::io::read_to_string(std::io::stdin())
                    .expect("unable to read problem from stdin")
            } else {
                std::fs::read_to_string(p)
                    .expect("unable to read problem file")
            }
        })
        .unwrap_or_else(|| htn_playing::EXAMPLE_PROBLEM.into());

    // SAFETY: only hold the stdin lock during planning
    let (end_state, plan) = {
        let domain = htn_playing::parse_domain(&domain_src)?;
        let mut problem = htn_playing::parse_problem(&problem_src)?;

        if let Some(path) = args.trace_file {
            let output: Box<dyn std::io::Write> = if path.as_os_str() == "-" {
                Box::new(std::io::stdout().lock())
            } else {
                Box::new(std::fs::File::create(path)
                    .expect("unable to open trace file"))
            };

            let tracer = Rc::new(RefCell::new(Tracer {
                output,
                stepped: args.stepped,
            }));

            problem.tracer = tracer;
        }

        domain.plan(problem)?
    };

    if let Some(p) = args.plan_file {
        if p.as_os_str() == "-" {
            println!("plan: {:?}", plan);
        } else {
            std::fs::write(p, format!("{:#?}", plan))?;
        }
    }

    if let Some(p) = args.final_state_file {
        if p.as_os_str() == "-" {
            println!("final state: {:?}", end_state.borrow().deref());
        } else {
            std::fs::write(p, format!("{:#?}", end_state.borrow().deref()))?;
        }
    }

    Ok(())
}
